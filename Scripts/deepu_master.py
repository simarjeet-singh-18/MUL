"""
DEEPU: Deeper Granular Within-Layer Machine Unlearning
(Vurity, Yan, Albanese — IEEE Trans. Information Forensics & Security, vol. 21, 2026)

Faithful reimplementation of the paper's core unlearning logic (Algorithm 1 +
Eqs. (1)-(7)), wired into the SAME experimental harness as the user's KD-based
unlearning script so a clean head-to-head comparison can be drawn.

WHAT IS HELD IDENTICAL TO THE ORIGINAL SCRIPT (so only the *method* differs)
---------------------------------------------------------------------------
  * Architectures        : resnet18 (cifar10) / resnet50 (cifar100,food101,places365),
                           with the same conv1(3x3,s1,p1)+maxpool=Identity CIFAR tweak
                           and fc -> num_classes head.
  * Starting weights      : the TEACHER checkpoint (TEACHER_PATH) — i.e. the *same*
                           pretrained "original model" M that the imp/nor students
                           are initialised from.
  * Data / loaders        : the same long-tailed train set and test set via utils
                           (called with pipeline="nor" so labels stay 0..num_classes-1).
  * Optimiser scale       : lr = 1e-3, batch_size = 32 (matches main() in the original).
  * Seeding / device      : the original setup() is reused verbatim.
  * Evaluation            : identical protocol — per-class accuracy on the test set and
                           "Retain Accuracy" = mean over all classes except forget_class
                           (plus we additionally print Forget-class accuracy, ACC_f,
                           which is DEEPU's headline metric and should collapse to ~0).

WHAT IS DELIBERATELY *NOT* MAPPED (no analogue in DEEPU — mapping them would
distort the paper's method):
  * temperature, KL distillation, num_epochs, head/mid/tail group rescaling.
    DEEPU is a one-shot weight edit + periodic stabilisation, not a training loop.

INTEGRATION NOTES
-----------------
  1. This script imports your `utils.data_loaders` / `utils.process_args`. It calls
     them with pipeline="nor" purely to obtain the FULL long-tailed train set and the
     standard test set with original labels (0..num_classes-1) and a num_classes head.
     If your utils has an explicit branch keyed on pipeline, just make sure "nor"
     returns the full train set (it already must, since imp/nor need every class).
  2. D_f / D_r are derived here from `train_loader.dataset` by filtering on the target
     label, so no change to your data pipeline is required.
  3. The unlearned model is saved next to your students as `deepu_im{imb}_cls{fc}.pth`.

Run, e.g.:
  python deepu.py --dataset cifar100 --imb-factor 100 --forget-class tail \
                  --clustering-type mean --seed 18 \
                  --deepu-alpha 2 --deepu-beta 2 \
                  --deepu-delta-inf 90 --deepu-delta-shared 50 --deepu-lambda 1.0
"""

import os
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import models

from utils import data_loaders, process_args


# ----------------------------------------------------------------------------- #
#  Seeding / device / forget-class resolution  (reused from the original script) #
# ----------------------------------------------------------------------------- #
def setup(dataset, forget_class, clustering_type, pipeline, SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Same cluster tables as the original script (only used to resolve the
    # "head"/"mid"/"tail" keyword into a concrete forget-class index, so that
    # DEEPU forgets exactly the SAME class the KD pipelines forget).
    clusters = {
        "cifar10": {
            "manual": {"head": list(range(3)), "mid": list(range(3, 7)), "tail": list(range(7, 10))},
            "mean":   {"head": [0, 1], "mid": [2, 3, 4, 7], "tail": [5, 6, 8, 9]},
            "mvm":    {"head": [0, 1], "mid": [2, 3, 4, 7], "tail": [5, 6, 8, 9]},
        },
        "cifar100": {
            "manual": {"head": list(range(0, 21)), "mid": list(range(21, 61)), "tail": list(range(61, 100))},
            "mean": {
                "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 28, 30, 33, 36, 39, 41, 48, 53],
                "mid":  [11, 18, 25, 26, 27, 29, 31, 32, 34, 35, 37, 38, 40, 42, 43, 47, 49, 51, 52, 54, 56, 57, 58, 60, 61, 62, 69, 76, 82],
                "tail": [44, 45, 46, 50, 55, 59, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            },
            "mvm": {
                "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 29, 30, 33, 36, 39, 41, 48, 53, 56, 60, 82],
                "mid":  [11, 25, 27, 31, 32, 34, 35, 37, 38, 40, 42, 43, 44, 47, 49, 51, 52, 54, 57, 58, 61, 62, 63, 67, 69, 74, 76, 94],
                "tail": [45, 46, 50, 55, 59, 64, 65, 66, 68, 70, 71, 72, 73, 75, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99],
            },
        },
        "food101": {
            "manual": {"head": list(range(0, 25)), "mid": list(range(25, 50)), "tail": list(range(50, 101))},
            "mean": {
                "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 38, 40, 44, 45, 54, 69],
                "mid":  [22, 26, 31, 36, 37, 39, 41, 42, 43, 46, 47, 48, 49, 51, 52, 53, 59, 60, 61, 63, 64, 65, 68, 70, 75, 81],
                "tail": [50, 55, 56, 57, 58, 62, 66, 67, 71, 72, 73, 74, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
            },
            "mvm": {
                "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 40, 41, 44, 45, 54, 63, 64, 69],
                "mid":  [26, 31, 37, 39, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 55, 59, 60, 61, 62, 65, 68, 70, 71, 75, 76, 78, 79, 81, 88, 91],
                "tail": [56, 57, 58, 66, 67, 72, 73, 74, 77, 80, 82, 83, 84, 85, 86, 87, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100],
            },
        },
        "places365": {
            "manual": {"head": list(range(0, 75)), "mid": list(range(75, 250)), "tail": list(range(250, 365))},
            "mean": {
                "head": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 24, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 40, 41, 42, 44, 45, 47, 48, 49, 55, 58, 59, 60, 61, 64, 65, 68, 70, 71, 72, 73, 74, 76, 82, 83, 86, 89, 95, 98, 105, 111, 112, 118, 135, 145, 152, 157],
                "mid":  [8, 12, 19, 20, 22, 23, 25, 28, 32, 38, 39, 43, 46, 50, 51, 52, 53, 54, 56, 57, 62, 63, 66, 69, 77, 78, 79, 81, 84, 85, 87, 88, 90, 92, 93, 94, 96, 100, 102, 104, 106, 107, 108, 110, 113, 116, 117, 119, 121, 122, 123, 124, 125, 126, 129, 131, 133, 136, 137, 140, 141, 142, 143, 144, 147, 149, 150, 151, 155, 156, 158, 159, 163, 164, 165, 166, 167, 168, 169, 170, 173, 175, 176, 177, 179, 180, 184, 189, 190, 191, 195, 199, 201, 207, 208, 225, 247, 251, 263, 276, 277],
                "tail": [i for i in range(365) if i not in
                         set([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 24, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 40, 41, 42, 44, 45, 47, 48, 49, 55, 58, 59, 60, 61, 64, 65, 68, 70, 71, 72, 73, 74, 76, 82, 83, 86, 89, 95, 98, 105, 111, 112, 118, 135, 145, 152, 157]
                                 + [8, 12, 19, 20, 22, 23, 25, 28, 32, 38, 39, 43, 46, 50, 51, 52, 53, 54, 56, 57, 62, 63, 66, 69, 77, 78, 79, 81, 84, 85, 87, 88, 90, 92, 93, 94, 96, 100, 102, 104, 106, 107, 108, 110, 113, 116, 117, 119, 121, 122, 123, 124, 125, 126, 129, 131, 133, 136, 137, 140, 141, 142, 143, 144, 147, 149, 150, 151, 155, 156, 158, 159, 163, 164, 165, 166, 167, 168, 169, 170, 173, 175, 176, 177, 179, 180, 184, 189, 190, 191, 195, 199, 201, 207, 208, 225, 247, 251, 263, 276, 277])],
            },
            "mvm": {
                "head": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 24, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 47, 48, 49, 51, 55, 57, 58, 59, 60, 61, 63, 64, 65, 68, 70, 71, 72, 73, 74, 76, 79, 82, 83, 86, 89, 95, 98, 100, 104, 105, 111, 112, 113, 118, 135, 145, 147, 152, 155, 157],
                "mid":  [8, 12, 19, 20, 22, 23, 25, 28, 32, 39, 43, 46, 50, 52, 53, 54, 56, 62, 66, 69, 77, 78, 81, 84, 85, 87, 88, 90, 92, 93, 94, 96, 102, 103, 106, 107, 108, 110, 116, 117, 119, 121, 122, 123, 124, 125, 126, 129, 131, 133, 136, 137, 140, 141, 142, 143, 144, 149, 150, 151, 154, 156, 158, 159, 163, 164, 165, 166, 167, 168, 169, 170, 173, 175, 176, 177, 179, 180, 184, 188, 189, 190, 191, 195, 199, 201, 207, 208, 211, 225, 247, 251, 255, 263, 276, 277, 294],
                "tail": [i for i in range(365) if i not in
                         set([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 24, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 47, 48, 49, 51, 55, 57, 58, 59, 60, 61, 63, 64, 65, 68, 70, 71, 72, 73, 74, 76, 79, 82, 83, 86, 89, 95, 98, 100, 104, 105, 111, 112, 113, 118, 135, 145, 147, 152, 155, 157]
                                 + [8, 12, 19, 20, 22, 23, 25, 28, 32, 39, 43, 46, 50, 52, 53, 54, 56, 62, 66, 69, 77, 78, 81, 84, 85, 87, 88, 90, 92, 93, 94, 96, 102, 103, 106, 107, 108, 110, 116, 117, 119, 121, 122, 123, 124, 125, 126, 129, 131, 133, 136, 137, 140, 141, 142, 143, 144, 149, 150, 151, 154, 156, 158, 159, 163, 164, 165, 166, 167, 168, 169, 170, 173, 175, 176, 177, 179, 180, 184, 188, 189, 190, 191, 195, 199, 201, 207, 208, 211, 225, 247, 251, 255, 263, 276, 277, 294])],
            },
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
            forget_class = first_common(
                clusters[dataset]["manual"][forget_class],
                clusters[dataset]["mean"][forget_class],
                clusters[dataset]["mvm"][forget_class],
            )
        else:
            forget_class = int(forget_class)
    forget_class = int(forget_class)

    print("Dataset      ", dataset)
    print("Clustering   ", clustering_type, "(only used to resolve the forget-class index)")
    print("Forget class ", forget_class)
    print("Pipeline      deepu")
    return device, forget_class


# --------------------------------------------------------------------------- #
#  Model builder  (same architecture as the original load_models, trainable)  #
# --------------------------------------------------------------------------- #
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
    # DEEPU needs gradients w.r.t. the pretrained weights, so keep grads ON.
    for p in model.parameters():
        p.requires_grad = True
    print("\nOriginal (teacher) model loaded as DEEPU input M")
    return model


# --------------------------------------------------------------------------- #
#  Forget / retain subsets from the existing long-tailed train set            #
# --------------------------------------------------------------------------- #
def get_targets(dataset_obj):
    if hasattr(dataset_obj, "targets") and dataset_obj.targets is not None:
        return np.asarray(dataset_obj.targets)
    if hasattr(dataset_obj, "samples"):          # ImageFolder fallback
        return np.asarray([s[1] for s in dataset_obj.samples])
    return np.asarray([int(dataset_obj[i][1]) for i in range(len(dataset_obj))])


def forget_retain_loaders(train_loader, forget_class, batch_size):
    ds = train_loader.dataset
    targets = get_targets(ds)
    forget_idx = np.where(targets == forget_class)[0].tolist()
    retain_idx = np.where(targets != forget_class)[0].tolist()
    print(f"|D_f| = {len(forget_idx)}   |D_r| = {len(retain_idx)}")

    forget_loader = DataLoader(Subset(ds, forget_idx), batch_size=batch_size,
                               shuffle=True, num_workers=0)
    retain_loader = DataLoader(Subset(ds, retain_idx), batch_size=batch_size,
                               shuffle=True, num_workers=0)
    return forget_loader, retain_loader


# --------------------------------------------------------------------------- #
#  DEEPU building blocks                                                        #
# --------------------------------------------------------------------------- #
def reference_gradients(model, loader, device, selected_names, max_batches=None):
    """Mean per-weight gradient of the CE loss over `loader` (Eq. 1 inputs g_f / g_r)."""
    model.eval()                                  # deterministic BN/dropout for scoring
    model.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        loss.backward()
        total += x.size(0)
        if max_batches is not None and (i + 1) >= max_batches:
            break
    total = max(total, 1)
    grads = {n: (p.grad.detach().clone() / total if p.grad is not None else torch.zeros_like(p))
             for n, p in model.named_parameters() if n in selected_names}
    model.zero_grad(set_to_none=True)
    return grads


def kmeans2(points, iters=20, seed=0):
    """Minimal k-means with k=2 on standardised 2-D points (w_i, SNR_i).
    Returns (distance_to_assigned_centroid) for each point."""
    n = points.shape[0]
    if n == 1:
        return torch.zeros(1, device=points.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.randperm(n, generator=g)[:2].to(points.device)
    centroids = points[init].clone()
    for _ in range(iters):
        d = torch.cdist(points, centroids)        # (n, 2)
        assign = d.argmin(dim=1)
        for k in range(2):
            m = assign == k
            if m.any():
                centroids[k] = points[m].mean(dim=0)
    d = torch.cdist(points, centroids)
    assign = d.argmin(dim=1)
    return d.gather(1, assign.view(-1, 1)).squeeze(1)


def _standardise(a, b):
    """Stack two 1-D tensors into (N,2) standardised features for clustering."""
    def z(t):
        return (t - t.mean()) / (t.std() + 1e-12)
    return torch.stack([z(a.float()), z(b.float())], dim=1)


def select_layers(model, alpha, beta):
    """L_u = first alpha + last beta weight tensors (Eq. 7).
    'weight tensors' = Conv/Linear .weight (ndim>=2); BN and biases are left alone."""
    weight_names = [n for n, p in model.named_parameters()
                    if n.endswith(".weight") and p.ndim >= 2]
    L = len(weight_names)
    a = max(0, min(alpha, L))
    b = max(0, min(beta, L))
    chosen = weight_names[:a] + weight_names[L - b:] if b > 0 else weight_names[:a]
    # de-dup while preserving order (in case alpha+beta overlap on tiny models)
    seen, ordered = set(), []
    for n in chosen:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    print(f"Total weight tensors L={L}; selected L_u (alpha={a}, beta={b}): {ordered}")
    return ordered


def stabilisation_step(model, retain_loader, optimizer, device, n_batches=1):
    """Eq. (6): theta_Lu <- theta_Lu - eta * grad L_r(theta_Lu). One (or n) retain batch."""
    model.eval()
    it = iter(retain_loader)
    criterion = nn.CrossEntropyLoss()
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def _apply_group_update(new_w_flat, w_flat, snr_flat, gfabs_flat, idx,
                        delta_pct, mode, lam, sigma, seed):
    """Apply the DEEPU update rule to one weight group (influential or shared)."""
    if idx.numel() == 0:
        return
    w_g = w_flat[idx]
    snr_g = snr_flat[idx]
    gf_g = gfabs_flat[idx]

    if idx.numel() >= 2:
        feats = _standardise(w_g, snr_g)
        dist = kmeans2(feats, seed=seed)
        thr = float(np.percentile(dist.detach().cpu().numpy(), delta_pct))
    else:
        dist = torch.zeros_like(w_g)
        thr = float("inf")

    noise = torch.randn_like(w_g) * sigma

    if mode == "influential":
        # Eq. (3): w' = (w + Gamma) * 1[d <= delta_inf]; i.e. reset far weights to 0,
        # perturb the near ones with N(0, sigma_inf^2).
        far = dist > thr
        vals = w_g + noise
        vals[far] = 0.0
    elif mode == "shared":
        # Eq. (4): decay near weights by exp(-lambda*|g_f|); perturb far ones.
        near = dist <= thr
        decay = w_g * torch.exp(-lam * gf_g)
        vals = torch.where(near, decay, w_g + noise)
    else:
        raise ValueError(mode)

    new_w_flat[idx] = vals


def deepu_unlearn(model, forget_loader, retain_loader, device, selected_names,
                  tau_low, tau_high, delta_inf, delta_shared, lam,
                  sigma_scale_inf, sigma_scale_shared, N_interval,
                  stab_lr, stab_batches, grad_batches, eps=1e-12):
    """One-shot DEEPU edit over the selected layers, per Algorithm 1."""
    print("\nComputing reference gradients g_f (forget) and g_r (retain)...")
    g_f = reference_gradients(model, forget_loader, device, selected_names, grad_batches)
    g_r = reference_gradients(model, retain_loader, device, selected_names, grad_batches)

    sel_params = [p for n, p in model.named_parameters() if n in selected_names]
    stab_opt = optim.SGD(sel_params, lr=stab_lr)   # plain SGD step, matching Eq. (6)

    params = dict(model.named_parameters())
    counter = 0

    for li, name in enumerate(selected_names):
        p = params[name]
        gf = g_f[name]
        gr = g_r[name]

        # Eq. (1): per-weight SNR = |g_f|^2 / (|g_r|^2 + eps)
        snr = (gf ** 2) / (gr ** 2 + eps)

        w_flat = p.data.view(-1)
        snr_flat = snr.view(-1)
        gfabs_flat = gf.abs().view(-1)

        # Eq. (2): categorise by SNR thresholds
        inf_idx = (snr_flat > tau_high).nonzero(as_tuple=False).squeeze(1)
        shared_idx = ((snr_flat >= tau_low) & (snr_flat <= tau_high)).nonzero(as_tuple=False).squeeze(1)
        # everything else (snr < tau_low) is non-influential -> left unchanged (Eq. 5)

        # sigma derived from this layer's forget-gradient statistics; sigma_inf > sigma_shared
        sigma_f = float(gfabs_flat.std().item()) + eps
        sigma_inf = sigma_scale_inf * sigma_f
        sigma_shared = sigma_scale_shared * sigma_f

        new_w_flat = w_flat.clone()
        _apply_group_update(new_w_flat, w_flat, snr_flat, gfabs_flat, inf_idx,
                            delta_inf, "influential", lam, sigma_inf, seed=li)
        _apply_group_update(new_w_flat, w_flat, snr_flat, gfabs_flat, shared_idx,
                            delta_shared, "shared", lam, sigma_shared, seed=li + 10000)
        p.data.copy_(new_w_flat.view_as(p.data))

        print(f"  [{name}] inf={inf_idx.numel()}  shared={shared_idx.numel()}  "
              f"non={snr_flat.numel() - inf_idx.numel() - shared_idx.numel()}  "
              f"sigma_f={sigma_f:.3e}")

        # Inter-layer stabilisation every N layer updates (Eq. 6)
        counter += 1
        if N_interval > 0 and (counter % N_interval == 0):
            stabilisation_step(model, retain_loader, stab_opt, device, stab_batches)

    # optional final stabilisation pass if the interval didn't land on the last layer
    if N_interval > 0 and (counter % N_interval != 0):
        stabilisation_step(model, retain_loader, stab_opt, device, stab_batches)

    return model


# --------------------------------------------------------------------------- #
#  Evaluation  (identical metric to the original train() eval block)          #
# --------------------------------------------------------------------------- #
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

    retain_acc, retain_classes = 0.0, 0
    forget_acc = 0.0
    for i in range(num_classes):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        print(f"{i}: {acc:.2f}%")
        if i == forget_class:
            forget_acc = acc
        else:
            retain_acc += acc
            retain_classes += 1
    retain_acc /= max(retain_classes, 1)

    print(f"\nForget-class accuracy (ACC_f, lower=better): {forget_acc:.2f}%")
    print(f"Retain Accuracy (mean over non-forget classes): {retain_acc:.2f}%")
    return forget_acc, retain_acc


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="DEEPU class-level machine unlearning")
    # --- shared with the original harness ---
    p.add_argument("--imb-factor", default="200")
    p.add_argument("--dataset", default="cifar10", help="cifar10 | cifar100 | food101 | places365")
    p.add_argument("--forget-class", default="head", help="head | mid | tail | numeric")
    p.add_argument("--clustering-type", default="manual", help="only used to resolve the forget-class index")
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # --- DEEPU hyperparameters (Algorithm 1) ---
    p.add_argument("--deepu-alpha", type=int, default=2, help="# initial weight tensors to edit")
    p.add_argument("--deepu-beta", type=int, default=2, help="# final weight tensors to edit (incl. fc)")
    p.add_argument("--deepu-tau-low", type=float, default=0.9, help="tau_low SNR threshold")
    p.add_argument("--deepu-tau-high", type=float, default=1.1, help="tau_high SNR threshold")
    p.add_argument("--deepu-delta-inf", type=float, default=90.0,
                   help="percentile: influential weights farther than this are hard-reset to 0")
    p.add_argument("--deepu-delta-shared", type=float, default=50.0,
                   help="percentile: shared weights nearer than this are exp-decayed")
    p.add_argument("--deepu-lambda", type=float, default=1.0, help="exponential decay factor lambda")
    p.add_argument("--deepu-sigma-inf", type=float, default=1.0, help="noise scale multiplier for influential")
    p.add_argument("--deepu-sigma-shared", type=float, default=0.5, help="noise scale multiplier for shared")
    p.add_argument("--deepu-N", type=int, default=1, help="stabilise after every N layer updates")
    p.add_argument("--deepu-stab-lr", type=float, default=0.001, help="eta for the stabilisation SGD step")
    p.add_argument("--deepu-stab-batches", type=int, default=1, help="# retain batches per stabilisation step")
    p.add_argument("--deepu-grad-batches", type=int, default=None,
                   help="cap batches used to estimate g_f/g_r (None = full pass; useful for places365)")
    return p.parse_args()


def deepu_save_path(student_save_dir, imb_factor, forget_class):
    """Reuse the original path scheme, swapping the pipeline token to 'deepu'."""
    for tok in ("nor_im", "imp_im", "orcl_im"):
        if tok in student_save_dir:
            return student_save_dir.replace(tok, "deepu_im", 1)
    d = os.path.dirname(student_save_dir)
    return os.path.join(d, f"deepu_im{imb_factor}_cls{forget_class}.pth")


def main():
    args = parse_args()

    learning_rate = 0.001
    batch_size = 32

    # Resolve device + forget-class index exactly as the KD pipelines do.
    device, forget_class = setup(args.dataset, args.forget_class,
                                 args.clustering_type, "deepu", args.seed)

    # Pull the SAME loaders/paths the KD pipelines use. pipeline="nor" gives the full
    # long-tailed train set + standard test set with original labels + num_classes head.
    (TRAIN_DATA, TEST_DATA, num_classes, num_epochs, TEACHER_PATH,
     STUDENT_SAVE_DIR, ORACLE_SAVE_PATH, P_AVG_PLOT_DIR, P_K_BAR_PLOT_DIR) = \
        process_args(args.dataset, args.imb_factor, args.forget_class,
                     args.clustering_type, "nor")

    train_loader, test_loader = data_loaders(
        args.dataset, TRAIN_DATA, TEST_DATA, batch_size, "nor", forget_class)

    # Build forget/retain views of the training data.
    forget_loader, retain_loader = forget_retain_loaders(train_loader, forget_class, batch_size)

    # Load the original model M (= teacher) and select the layers to edit.
    model = build_model(args.dataset, device, num_classes, TEACHER_PATH)
    selected = select_layers(model, args.deepu_alpha, args.deepu_beta)

    # --- run DEEPU ---
    t0 = time.perf_counter()
    model = deepu_unlearn(
        model, forget_loader, retain_loader, device, selected,
        tau_low=args.deepu_tau_low, tau_high=args.deepu_tau_high,
        delta_inf=args.deepu_delta_inf, delta_shared=args.deepu_delta_shared,
        lam=args.deepu_lambda,
        sigma_scale_inf=args.deepu_sigma_inf, sigma_scale_shared=args.deepu_sigma_shared,
        N_interval=args.deepu_N, stab_lr=args.deepu_stab_lr,
        stab_batches=args.deepu_stab_batches, grad_batches=args.deepu_grad_batches,
    )
    dt = time.perf_counter() - t0
    print(f"\nDEEPU unlearning wall-clock: {dt:.4f} s")

    # --- evaluate with the identical protocol ---
    print("\n==== Post-unlearning evaluation ====")
    evaluate(model, test_loader, device, num_classes, forget_class)

    # --- save next to the KD students ---
    save_path = deepu_save_path(STUDENT_SAVE_DIR, args.imb_factor, forget_class)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved unlearned model to: {save_path}")


if __name__ == "__main__":
    main()