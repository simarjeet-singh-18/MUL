"""
L-CODEC (CR variant): Deep Unlearning via Randomized Conditionally Independent Hessians
(Mehta, Pal, Singh, Ravi — CVPR 2022)

Self-contained, faithful reimplementation of the authors' released pipeline
(vsingh-group/LCODEC-deep-unlearning), CR ("naive-Newton / certified-removal")
update variant, wired into the SAME experimental harness as the user's KD script
and the DEEPU / SalUn / DELETE baselines so methods differ ONLY in the mechanism.

WHY THE CR VARIANT
------------------
The faithful "Sekhari" update (paper Eq. 15) needs gradients/params checkpointed
from the last two training epochs of the ORIGINAL run (getOldPandG in the repo).
Those are training-time artifacts that don't exist here. The authors' code ships a
second, self-contained update -- CR_NaiveNewton -- that uses the SAME L-FOCI
selection and the SAME finite-difference *sample* Hessian, but no stored old-Hessian
and no DP noise:   w'_P = w_P + H_P^{-1} g_P   on the selected block only.
This runs today with no training-time artifacts and is a genuine variant from the
released code (params.HessType == 'CR'), not a mislabel.

FAITHFULLY PORTED FROM THE AUTHORS' REPO
----------------------------------------
  * OneNN_Torch, codec2, codec3            (codec/neighbors.py, codec/torch_codec.py)
  * foci / cheap_foci                       (codec/torch_foci.py)
  * ActivationsHook  (Linear: mean over batch; Conv: mean over batch+spatial)
                                            (scrub/hypercolumn.py)
  * input perturbation loop  x + noise*randn, collect (activations, losses)
                                            (scrub/scrub_tools.py inp_perturb)
  * reverseLinearIndexingToLayers, getVectorizedGrad, updateModelParams
                                            (scrub/scrub_tools.py, scrub/grad_utils.py)
  * finite-difference Hessian (FD): outer(dg) / ||dw||_1
                                            (scrub/grad_utils.py getHessian)
  * CR_NaiveNewton: w + solve(H + l2*I, g)  (scrub/scrub_tools.py)

ONE DELIBERATE, DISCLOSED DEVIATION
-----------------------------------
The released FD line reuses sampGrad1 for the "stepped" gradient, so its
finite-difference Hessian collapses to ZERO (an apparent bug). We implement the
paper's INTENDED Eq.-15 finite difference instead: grad at w vs grad at w after one
SGD step (fd_lr), params w1 vs w2. Reproducing the zero-Hessian bug would make the
method meaningless.

ADAPTATION TO CLASS FORGETTING
------------------------------
The paper unlearns one sample at a time; the repo also supports batch-level scrubbing
(scrub_batch_size) where the hooks average activations over the batch. For class
forgetting we iterate over batches of the forget class, applying the select+scrub
update per batch (each batch is the "z'"). Cap with --lcodec-max-batches.

HELD IDENTICAL TO THE OTHER BASELINES
-------------------------------------
  Architecture (resnet18/50 + CIFAR conv1/maxpool tweak + fc->num_classes), teacher
  checkpoint theta_o, loaders, forget-class resolution, seeding, and the evaluation
  protocol are the same as DEEPU / SalUn / DELETE and consistent with the KD harness.

USAGE (reuses your config matrix; extra --deepu-*/--salun-*/--delete-* flags ignored)
------------------------------------------------------------------------------------
  python lcodec.py --dataset cifar10 --imb-factor 100 --forget-class head \
                   --clustering-type manual --seed 18 \
                   --lcodec-perturbations 100 --lcodec-selection foci \
                   --lcodec-l2 0.01 --lcodec-fd-lr 1e-3
"""

import os
import copy
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector as p2v
from torchvision import models

from utils import data_loaders, process_args


# =========================================================================== #
#  Ported CODEC / FOCI core  (codec/neighbors.py, torch_codec.py, torch_foci.py)
# =========================================================================== #
def OneNN_Torch(X, p=2):
    if X.dim() == 1:
        X = X.view(-1, 1)
    pdists = torch.cdist(X, X, p=p, compute_mode='use_mm_for_euclid_dist_if_necessary')
    pdists.fill_diagonal_(float('inf'))
    return torch.argmin(pdists, dim=1)


def codec2(Z, Y):
    """Chatterjee/CODEC dependence of Y on Z (unconditional)."""
    if Z.dim() == 1:
        Z = Z.reshape(-1, 1)
    if Y.dim() == 2 and Y.shape[1] == 1:
        Y = Y.squeeze()
    n, q = Z.shape
    M = OneNN_Torch(Z)
    p = torch.argsort(Y)
    R = torch.arange(n, device=Z.device)
    tmpR = torch.arange(n, device=Z.device)
    R[p] = tmpR + 1
    RM = R[M]
    minRM = n * torch.minimum(R, RM)
    L = (n + 1) - R
    return (minRM - L ** 2).sum() / torch.dot(L.float(), (n - L).float())


def codec3(Z, Y, X):
    """CODEC dependence of Y on Z given X (conditional)."""
    if Z.dim() == 1:
        Z = Z.view(-1, 1)
    if X.dim() == 1:
        X = X.view(-1, 1)
    if Y.dim() == 2 and Y.shape[1] == 1:
        Y = Y.squeeze()
    n, px = X.shape
    N = OneNN_Torch(X)
    W = torch.hstack((X, Z))
    M = OneNN_Torch(W)
    p = torch.argsort(Y)
    R = torch.arange(n, device=Z.device)
    tmpR = torch.arange(n, device=Z.device)
    R[p] = tmpR + 1
    RM = R[M]
    RN = R[N]
    minRM = torch.minimum(R, RM)
    minRN = torch.minimum(R, RN)
    denom = (R - minRN).sum()
    if denom == 0:
        return torch.tensor(-100.0, device=Z.device)
    return (minRM - minRN).sum().float() / denom.float()


def foci(X, Y, earlyStop=True, verbose=False):
    """L-FOCI feature ordering: greedily build the sufficient (Markov-blanket) set."""
    p = X.shape[1]
    maxval = -100.0
    maxind = None
    for i in range(p):
        tmp = codec2(X[:, i], Y)
        if tmp > maxval:
            maxval = tmp
            maxind = i
    all_inds = np.arange(p)
    deplist = [maxind]
    depset = set(deplist)
    indepset = set(all_inds).difference(depset)
    indeplist = list(indepset)
    ordering = [maxind]
    scores = [float(maxval)]
    for k in range(p - 1):
        cX = X[:, deplist]
        maxval = -100.0
        mostdepL = None
        for l in indeplist:
            tmp = codec3(X[:, l], Y, cX)
            if tmp > maxval:
                maxval = tmp
                mostdepL = l
        if mostdepL is None:
            break
        if maxval <= 0.0 and earlyStop:
            break
        depset.add(mostdepL)
        indepset.remove(mostdepL)
        deplist.append(mostdepL)
        indeplist = list(indepset)
        ordering.append(mostdepL)
        scores.append(float(maxval))
    return ordering, scores


def cheap_foci(X, Y):
    """Top-1 most-dependent slice only (fast path for very wide layers)."""
    p = X.shape[1]
    maxval = -100.0
    maxind = None
    for i in range(p):
        tmp = codec2(X[:, i], Y)
        if tmp > maxval:
            maxval = tmp
            maxind = i
    return [maxind], [float(maxval)]


# =========================================================================== #
#  Activation hooks  (scrub/hypercolumn.py ActivationsHook)                    #
# =========================================================================== #
class ActivationsHook(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.layers = []
        self.activations = []
        self.hooks = []
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                self.hooks.append(m.register_forward_hook(self._linear_hook))
                self.layers.append(m)
            elif isinstance(m, nn.Conv2d):
                self.hooks.append(m.register_forward_hook(self._conv_hook))
                self.layers.append(m)

    def getLayers(self):
        return self.layers

    def _linear_hook(self, module, inp, out):
        self.activations.append(out.mean(dim=[0]))          # mean over batch

    def _conv_hook(self, module, inp, out):
        self.activations.append(out.mean(dim=[0, 2, 3]))    # mean over batch + spatial

    def getActivations(self, x):
        self.activations = []
        out = self.model(x)
        return self.activations, out

    def clearHooks(self):
        for h in self.hooks:
            h.remove()


# =========================================================================== #
#  Slice<->layer indexing, grad extraction, update  (scrub_tools / grad_utils) #
# =========================================================================== #
def reverseLinearIndexingToLayers(selectedSlices, torchLayers):
    ind_list = []
    for myslice in selectedSlices:
        prevslicecnt = 0
        if isinstance(torchLayers[0], nn.Conv2d):
            nextslicecnt = torchLayers[0].out_channels
        else:
            nextslicecnt = torchLayers[0].out_features
        for l in range(len(torchLayers)):
            if myslice < nextslicecnt:
                modslice = myslice - prevslicecnt
                ind_list.append([torchLayers[l], int(modslice)])
                break
            prevslicecnt = nextslicecnt
            nxt = torchLayers[l + 1]
            nextslicecnt += nxt.out_channels if isinstance(nxt, nn.Conv2d) else nxt.out_features
    return ind_list


def getGradObjs(model):
    grad_objs = {}
    for module in model.modules():
        for (name, param) in module.named_parameters(recurse=False):
            grad_objs[(str(module), name)] = param.grad
    return grad_objs


def getVectorizedGrad(gradlist, slices_to_update, device):
    mapDict = {}
    vect_grad = torch.zeros(0, device=device)
    vect_param = torch.zeros(0, device=device)
    for [layer, sliceID] in slices_to_update:
        for (pname, ptensor) in layer.named_parameters(recurse=False):
            orig_shape = ptensor[sliceID].shape
            vparam = torch.flatten(ptensor[sliceID])
            pgrad = gradlist[(str(layer), pname)]
            vgrad = torch.flatten(pgrad[sliceID])
            start = vect_grad.shape[0]
            vect_grad = torch.cat([vect_grad, vgrad], dim=0)
            vect_param = torch.cat([vect_param, vparam], dim=0)
            end = vect_grad.shape[0]
            mapDict[(str(layer), pname, sliceID)] = [start, end, orig_shape, ptensor]
    return vect_grad, vect_param, mapDict


def updateModelParams(updatedParams, reversalDict, model):
    for key in reversalDict.keys():
        start, end, orig_shape, param = reversalDict[key]
        _, _, sliceID = key
        vec_w = updatedParams[start:end]
        param[sliceID] = vec_w.reshape(orig_shape).clone().detach()


def getHessian_FD(dw1, dw2, w1, w2, hessian_device='cpu'):
    """Finite-difference Hessian: outer(dg) / ||dw||_1   (grad_utils.getHessian, FD)."""
    dw1 = dw1.to(hessian_device); dw2 = dw2.to(hessian_device)
    w1 = w1.to(hessian_device);   w2 = w2.to(hessian_device)
    grad_diff_outer = torch.einsum('p,q->pq', (dw1 - dw2), (dw1 - dw2))
    pdist = nn.PairwiseDistance(p=1)
    weight_scaling = pdist(w1.view(1, -1), w2.view(1, -1))
    return torch.div(grad_diff_outer, weight_scaling)


def CR_NaiveNewton(weight, grad, hessian, l2lambda=0.01, hessian_device='cpu'):
    """w' = w + (H + l2 I)^{-1} g   (scrub_tools.CR_NaiveNewton)."""
    original_device = weight.device
    H = hessian.to(hessian_device) + l2lambda * torch.eye(hessian.shape[0], device=hessian_device)
    newton = torch.linalg.solve(H, grad.to(hessian_device)).to(original_device)
    return weight + newton     # removal = positive-gradient (ascent) direction


# =========================================================================== #
#  One scrub step over a forget batch  (adapted inp_perturb, CR path)          #
# =========================================================================== #
def lcodec_scrub_batch(model, x, y, criterion, device, n_perturbations, noise_std,
                       selection, l2lambda, fd_lr, hessian_device, max_slice_params):
    # ---- 1. input perturbation -> (activations, losses) ----
    hook = ActivationsHook(model)
    torch_layers = hook.getLayers()
    acts_list, losses = [], []
    model.eval()
    for _ in range(n_perturbations):
        tmp = x + noise_std * torch.randn_like(x)
        acts, out = hook.getActivations(tmp)
        loss = criterion(out, y)
        acts_list.append(p2v(acts).detach())
        losses.append(loss.detach())
    acts = torch.vstack(acts_list)                     # (m, P_slices)
    losses = torch.stack(losses).to(device)            # (m,)
    hook.clearHooks()

    # ---- 2. slice selection ----
    P = acts.shape[1]
    if selection == 'full':
        selected = list(range(P))
    elif selection == 'one':
        selected = [int(np.random.permutation(P)[0])]
    elif selection == 'random':
        sel_full, _ = foci(acts, losses, earlyStop=True)
        selected = np.random.permutation(P)[:max(1, len(sel_full))].tolist()
    elif selection == 'cheap':
        selected, _ = cheap_foci(acts, losses)
    else:  # 'foci'
        selected, _ = foci(acts, losses, earlyStop=True)
    slices_to_update = reverseLinearIndexingToLayers(selected, torch_layers)

    # ---- 3. sample gradient at w  (vectGrad1, vectParams1) ----
    model.eval()
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss_before = criterion(out, y)
    loss_before.backward()
    grads1 = getGradObjs(model)
    vectGrad1, vectParams1, revIdx = getVectorizedGrad(grads1, slices_to_update, device)
    model.zero_grad(set_to_none=True)

    n_sel_params = vectGrad1.shape[0]
    if n_sel_params == 0:
        return float(loss_before.item()), 0
    if n_sel_params > max_slice_params:
        # Newton solve is O(k^3); guard pathological selections
        print(f"    [skip] selected block too large ({n_sel_params} > {max_slice_params} params)")
        return float(loss_before.item()), 0

    # ---- 4. grad at w' after one SGD step  (paper Eq.15 finite difference) ----
    model_copy = copy.deepcopy(model)
    layers_copy = [m for m in model_copy.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    slices_copy = reverseLinearIndexingToLayers(selected, layers_copy)
    opt_copy = optim.SGD(model_copy.parameters(), lr=fd_lr)
    model_copy.eval()
    out_c = model_copy(x); loss_c = criterion(out_c, y)
    opt_copy.zero_grad(); loss_c.backward(); opt_copy.step()          # one step -> w'
    out_c = model_copy(x); loss_c = criterion(out_c, y)
    opt_copy.zero_grad(); loss_c.backward()                          # grad at w'
    grads2 = getGradObjs(model_copy)
    vectGrad2, vectParams2, _ = getVectorizedGrad(grads2, slices_copy, device)

    # ---- 5. finite-difference Hessian + CR naive-Newton update ----
    H = getHessian_FD(vectGrad1, vectGrad2, vectParams1, vectParams2, hessian_device)
    updated = CR_NaiveNewton(vectParams1, vectGrad1, H, l2lambda=l2lambda,
                             hessian_device=hessian_device)

    with torch.no_grad():
        updateModelParams(updated, revIdx, model)
    model.zero_grad(set_to_none=True)
    return float(loss_before.item()), n_sel_params


# =========================================================================== #
#  Harness: setup / build_model / get_targets / evaluate  (as other baselines) #
# =========================================================================== #
def setup(dataset, forget_class, clustering_type, pipeline, SEED):
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    clusters = {
        "cifar10": {
            "manual": {"head": list(range(3)), "mid": list(range(3, 7)), "tail": list(range(7, 10))},
            "mean":   {"head": [0, 1], "mid": [2, 3, 4, 7], "tail": [5, 6, 8, 9]},
            "mvm":    {"head": [0, 1], "mid": [2, 3, 4, 7], "tail": [5, 6, 8, 9]},
        },
        "cifar100": {
            "manual": {"head": list(range(0, 21)), "mid": list(range(21, 61)), "tail": list(range(61, 100))},
            "mean": {"head": [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,19,20,21,22,23,24,28,30,33,36,39,41,48,53],
                     "mid":  [11,18,25,26,27,29,31,32,34,35,37,38,40,42,43,47,49,51,52,54,56,57,58,60,61,62,69,76,82],
                     "tail": [44,45,46,50,55,59,63,64,65,66,67,68,70,71,72,73,74,75,77,78,79,80,81,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]},
            "mvm": {"head": [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,26,28,29,30,33,36,39,41,48,53,56,60,82],
                    "mid":  [11,25,27,31,32,34,35,37,38,40,42,43,44,47,49,51,52,54,57,58,61,62,63,67,69,74,76,94],
                    "tail": [45,46,50,55,59,64,65,66,68,70,71,72,73,75,77,78,79,80,81,83,84,85,86,87,88,89,90,91,92,93,95,96,97,98,99]},
        },
        "food101": {
            "manual": {"head": list(range(0, 25)), "mid": list(range(25, 50)), "tail": list(range(50, 101))},
            "mean": {"head": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,27,28,29,30,32,33,34,35,38,40,44,45,54,69],
                     "mid":  [22,26,31,36,37,39,41,42,43,46,47,48,49,51,52,53,59,60,61,63,64,65,68,70,75,81],
                     "tail": [50,55,56,57,58,62,66,67,71,72,73,74,76,77,78,79,80,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]},
            "mvm": {"head": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,32,33,34,35,36,38,40,41,44,45,54,63,64,69],
                    "mid":  [26,31,37,39,42,43,46,47,48,49,50,51,52,53,55,59,60,61,62,65,68,70,71,75,76,78,79,81,88,91],
                    "tail": [56,57,58,66,67,72,73,74,77,80,82,83,84,85,86,87,89,90,92,93,94,95,96,97,98,99,100]},
        },
        "places365": {
            "manual": {"head": list(range(0, 75)), "mid": list(range(75, 250)), "tail": list(range(250, 365))},
            "mean": {"head": [0,1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,24,26,27,29,30,31,33,34,35,36,37,40,41,42,44,45,47,48,49,55,58,59,60,61,64,65,68,70,71,72,73,74,76,82,83,86,89,95,98,105,111,112,118,135,145,152,157],
                     "mid": [8,12,19,20,22,23,25,28,32,38,39,43,46,50,51,52,53,54,56,57,62,63,66,69,77,78,79,81,84,85,87,88,90,92,93,94,96,100,102,104,106,107,108,110,113,116,117,119,121,122,123,124,125,126,129,131,133,136,137,140,141,142,143,144,147,149,150,151,155,156,158,159,163,164,165,166,167,168,169,170,173,175,176,177,179,180,184,189,190,191,195,199,201,207,208,225,247,251,263,276,277],
                     "tail": [i for i in range(365) if i not in set([0,1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,24,26,27,29,30,31,33,34,35,36,37,40,41,42,44,45,47,48,49,55,58,59,60,61,64,65,68,70,71,72,73,74,76,82,83,86,89,95,98,105,111,112,118,135,145,152,157]+[8,12,19,20,22,23,25,28,32,38,39,43,46,50,51,52,53,54,56,57,62,63,66,69,77,78,79,81,84,85,87,88,90,92,93,94,96,100,102,104,106,107,108,110,113,116,117,119,121,122,123,124,125,126,129,131,133,136,137,140,141,142,143,144,147,149,150,151,155,156,158,159,163,164,165,166,167,168,169,170,173,175,176,177,179,180,184,189,190,191,195,199,201,207,208,225,247,251,263,276,277])]},
            "mvm": {"head": [0,1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,24,26,27,29,30,31,33,34,35,36,37,38,40,41,42,44,45,47,48,49,51,55,57,58,59,60,61,63,64,65,68,70,71,72,73,74,76,79,82,83,86,89,95,98,100,104,105,111,112,113,118,135,145,147,152,155,157],
                    "mid": [8,12,19,20,22,23,25,28,32,39,43,46,50,52,53,54,56,62,66,69,77,78,81,84,85,87,88,90,92,93,94,96,102,103,106,107,108,110,116,117,119,121,122,123,124,125,126,129,131,133,136,137,140,141,142,143,144,149,150,151,154,156,158,159,163,164,165,166,167,168,169,170,173,175,176,177,179,180,184,188,189,190,191,195,199,201,207,208,211,225,247,251,255,263,276,277,294],
                    "tail": [i for i in range(365) if i not in set([0,1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,24,26,27,29,30,31,33,34,35,36,37,38,40,41,42,44,45,47,48,49,51,55,57,58,59,60,61,63,64,65,68,70,71,72,73,74,76,79,82,83,86,89,95,98,100,104,105,111,112,113,118,135,145,147,152,155,157]+[8,12,19,20,22,23,25,28,32,39,43,46,50,52,53,54,56,62,66,69,77,78,81,84,85,87,88,90,92,93,94,96,102,103,106,107,108,110,116,117,119,121,122,123,124,125,126,129,131,133,136,137,140,141,142,143,144,149,150,151,154,156,158,159,163,164,165,166,167,168,169,170,173,175,176,177,179,180,184,188,189,190,191,195,199,201,207,208,211,225,247,251,255,263,276,277,294])]},
        },
    }

    def first_common(a, b, c):
        sb, sc = set(b), set(c)
        fc = next((x for x in a if x in sb and x in sc), None)
        if fc is None:
            raise ValueError("No common element found in clusters")
        return fc

    if isinstance(forget_class, str):
        if forget_class in ("head", "mid", "tail"):
            forget_class = first_common(clusters[dataset]["manual"][forget_class],
                                        clusters[dataset]["mean"][forget_class],
                                        clusters[dataset]["mvm"][forget_class])
        else:
            forget_class = int(forget_class)
    forget_class = int(forget_class)

    print("Dataset      ", dataset)
    print("Clustering   ", clustering_type, "(only used to resolve the forget-class index)")
    print("Forget class ", forget_class)
    print("Pipeline     ", pipeline)
    return device, forget_class


def build_model(dataset, device, num_classes, TEACHER_PATH):
    if dataset == "cifar10":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif dataset in ["cifar100", "food101", "places365"]:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    if dataset in ["cifar10", "cifar100"]:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = True
    print("\nOriginal (teacher) model theta_o loaded")
    return model


def get_targets(dataset_obj):
    if hasattr(dataset_obj, "targets") and dataset_obj.targets is not None:
        return np.asarray(dataset_obj.targets)
    if hasattr(dataset_obj, "samples"):
        return np.asarray([s[1] for s in dataset_obj.samples])
    return np.asarray([int(dataset_obj[i][1]) for i in range(len(dataset_obj))])


def forget_loader_from_train(train_loader, forget_class, batch_size):
    ds = train_loader.dataset
    targets = get_targets(ds)
    forget_idx = np.where(targets == forget_class)[0].tolist()
    print(f"|D_f| = {len(forget_idx)}   (L-CODEC scrubs forget-class batches)")
    return DataLoader(Subset(ds, forget_idx), batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def evaluate(model, test_loader, device, num_classes, forget_class):
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        _, pred = torch.max(model(x), 1)
        for i in range(y.size(0)):
            lbl = y[i].item()
            class_total[lbl] += 1
            if pred[i] == y[i]:
                class_correct[lbl] += 1
    retain_acc, retain_classes, forget_acc = 0.0, 0, 0.0
    for i in range(num_classes):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        print(f"{i}: {acc:.2f}%")
        if i == forget_class:
            forget_acc = acc
        else:
            retain_acc += acc; retain_classes += 1
    retain_acc /= max(retain_classes, 1)
    print(f"\nForget-class accuracy (ACC_f, lower=better): {forget_acc:.2f}%")
    print(f"Retain Accuracy (mean over non-forget classes): {retain_acc:.2f}%")
    return forget_acc, retain_acc


# =========================================================================== #
def parse_args():
    p = argparse.ArgumentParser(description="L-CODEC (CR variant) class-level machine unlearning")
    p.add_argument("--imb-factor", default="200")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--forget-class", default="head", help="head | mid | tail | numeric")
    p.add_argument("--clustering-type", default="manual", help="only used to resolve the forget-class index")
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # L-CODEC hyperparameters
    p.add_argument("--lcodec-perturbations", type=int, default=100, help="m: input perturbations for L-FOCI")
    p.add_argument("--lcodec-noise-std", type=float, default=0.1, help="input perturbation std (repo: 0.1)")
    p.add_argument("--lcodec-selection", default="foci", choices=["foci", "cheap", "random", "one", "full"])
    p.add_argument("--lcodec-l2", type=float, default=0.01, help="l2 smoothing on Hessian before inverse")
    p.add_argument("--lcodec-fd-lr", type=float, default=1e-3, help="SGD step for finite-difference Hessian")
    p.add_argument("--lcodec-batch-size", type=int, default=32)
    p.add_argument("--lcodec-max-batches", type=int, default=None, help="cap number of forget batches scrubbed")
    p.add_argument("--lcodec-hessian-device", default="cpu", help="'cpu' or 'cuda' for the Newton solve")
    p.add_argument("--lcodec-max-slice-params", type=int, default=20000,
                   help="skip a batch if its selected block exceeds this many params (Newton is O(k^3))")
    p.add_argument("--csv", default="/export/home/achyut/Simarjeet/MUL/Schedulers/food101/unlearning_results.csv",
                   help="append a results row to this CSV (shared schema across methods; '' disables)")
    return p.parse_known_args()


# Shared, method-agnostic results schema. Every baseline can append to the same
# file so results accumulate into one table (method column distinguishes rows).
CSV_FIELDS = [
    "timestamp", "method", "dataset", "imb_factor", "forget_class",
    "clustering_type", "seed", "forget_acc", "retain_acc", "ua",
    "wall_time_s", "hyperparams", "save_path",
]


def append_result_csv(csv_path, row):
    """Append one result row, writing the header if the file is new/empty."""
    if not csv_path:
        return
    import csv
    d = os.path.dirname(csv_path)
    if d:
        os.makedirs(d, exist_ok=True)
    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    print(f"Appended result to {csv_path}")


def lcodec_save_path(student_save_dir, imb_factor, forget_class):
    for tok in ("nor_im", "imp_im", "orcl_im", "deepu_im", "salun_im", "delete_im"):
        if tok in student_save_dir:
            return student_save_dir.replace(tok, "lcodec_im", 1)
    d = os.path.dirname(student_save_dir)
    return os.path.join(d, f"lcodec_im{imb_factor}_cls{forget_class}.pth")


def main():
    args, unknown = parse_args()
    if unknown:
        print(f"[note] ignoring unrelated args: {unknown}")

    batch_size = args.lcodec_batch_size
    device, forget_class = setup(args.dataset, args.forget_class,
                                 args.clustering_type, "lcodec", args.seed)

    (TRAIN_DATA, TEST_DATA, num_classes, num_epochs, TEACHER_PATH,
     STUDENT_SAVE_DIR, ORACLE_SAVE_PATH, P_AVG_PLOT_DIR, P_K_BAR_PLOT_DIR) = \
        process_args(args.dataset, args.imb_factor, args.forget_class, args.clustering_type, "nor")

    train_loader, test_loader = data_loaders(
        args.dataset, TRAIN_DATA, TEST_DATA, batch_size, "nor", forget_class)

    model = build_model(args.dataset, device, num_classes, TEACHER_PATH)
    forget_loader = forget_loader_from_train(train_loader, forget_class, batch_size)
    criterion = nn.CrossEntropyLoss()

    # --- L-CODEC (CR) scrub over forget-class batches ---
    t0 = time.perf_counter()
    n_done = 0
    for bidx, (x, y) in enumerate(forget_loader):
        if args.lcodec_max_batches is not None and bidx >= args.lcodec_max_batches:
            break
        x, y = x.to(device), y.to(device)
        loss_b, k = lcodec_scrub_batch(
            model, x, y, criterion, device,
            n_perturbations=args.lcodec_perturbations, noise_std=args.lcodec_noise_std,
            selection=args.lcodec_selection, l2lambda=args.lcodec_l2, fd_lr=args.lcodec_fd_lr,
            hessian_device=args.lcodec_hessian_device, max_slice_params=args.lcodec_max_slice_params,
        )
        n_done += 1
        print(f"  batch {bidx+1}: forget-loss-before={loss_b:.4f}, scrubbed {k} params")
    dt = time.perf_counter() - t0
    print(f"\nL-CODEC(CR) scrubbed {n_done} forget batches in {dt:.2f} s")

    print("\n==== Post-unlearning evaluation ====")
    forget_acc, retain_acc = evaluate(model, test_loader, device, num_classes, forget_class)
    print(f"Unlearning Accuracy (UA = 100 - ACC_f): {100.0 - forget_acc:.2f}%")

    save_path = lcodec_save_path(STUDENT_SAVE_DIR, args.imb_factor, forget_class)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved unlearned model to: {save_path}")

    # --- append a row to the shared results CSV ---
    import datetime
    hyper = (f"selection={args.lcodec_selection};perturb={args.lcodec_perturbations};"
             f"noise_std={args.lcodec_noise_std};l2={args.lcodec_l2};fd_lr={args.lcodec_fd_lr};"
             f"batch_size={batch_size};max_batches={args.lcodec_max_batches};batches_scrubbed={n_done}")
    append_result_csv(args.csv, {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "method": "lcodec_cr",
        "dataset": args.dataset,
        "imb_factor": args.imb_factor,
        "forget_class": forget_class,
        "clustering_type": args.clustering_type,
        "seed": args.seed,
        "forget_acc": f"{forget_acc:.2f}",
        "retain_acc": f"{retain_acc:.2f}",
        "ua": f"{100.0 - forget_acc:.2f}",
        "wall_time_s": f"{dt:.2f}",
        "hyperparams": hyper,
        "save_path": save_path,
    })


if __name__ == "__main__":
    main()