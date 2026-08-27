"""
SalUn: Saliency Unlearning
(Fan, Liu, Zhang, Wong, Wei, Liu — ICLR 2024,
 "Empowering Machine Unlearning via Gradient-Based Weight Saliency")

Self-contained, faithful reimplementation of SalUn's core classification method
(Algorithm 1, Eqs. (3)-(6)), wired into the SAME experimental harness as the
user's KD-based unlearning script and the DEEPU baseline, so all methods differ
ONLY in the unlearning mechanism. This file has NO dependency on deepu.py.

CORE METHOD (exactly as in the paper)
-------------------------------------
1. Gradient-based weight saliency mask (Eq. 3 + Eq. 5, classification uses CE):
       g_S = grad_theta  E_{(x,y)~D_f}[ CE(theta; x,y) ] |_{theta=theta_o}
       m_S = 1( |g_S| >= gamma )
   gamma is the sparsity-percentile of |g_S| over the ENTIRE parameter vector.
   The paper uses the median (== 50% sparsity) by default; a sparsity ratio s
   means s-fraction of weights are left unchanged, i.e. gamma = percentile(|g_S|,
   100*s), mask=1 where |g_S| >= gamma.
       - higher sparsity  -> fewer weights edited -> under-forgetting
       - lower  sparsity  -> more  weights edited -> over-forgetting

2. Saliency-masked Random Labeling (Eq. 4 + Eq. 6, Algorithm 1):
   Relabel the forget set ONCE with random labels y' != y  ->  D_f'.
   Fine-tune on D' = D_f' U D_r minimizing
       E_{D_f'}[ CE(theta_u; x,y') ]  +  alpha * E_{D_r}[ CE(theta_u; x,y) ]
   where theta_u = m_S ⊙ (Δθ + theta_o) + (1 - m_S) ⊙ theta_o (Eq. 4),
   implemented via masked-gradient SGD:
       theta_u <- theta_u - eta * ( m_S ⊙ g )
   so only salient weights move; non-salient weights stay frozen at theta_o.

HELD IDENTICAL TO THE ORIGINAL SCRIPT / DEEPU (so only the method differs)
-------------------------------------------------------------------------
  * Architecture (resnet18/50 + CIFAR conv1/maxpool tweak + fc->num_classes),
    teacher checkpoint (theta_o), loaders, D_f/D_r split, seeding, and the
    evaluation protocol (per-class acc + Retain Accuracy over non-forget classes)
    are the SAME as in the DEEPU run and consistent with the KD harness.
  * batch_size = 32 (matches the original main()).

USAGE (reuses your existing config matrix; extra --deepu-* flags are ignored)
-----------------------------------------------------------------------------
  python salun.py --dataset cifar10 --imb-factor 100 --forget-class head \
                  --clustering-type manual --seed 18 \
                  --salun-sparsity 0.5 --salun-epochs 10 --salun-lr 0.01 --salun-alpha 1.0
"""

import os
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models

from utils import data_loaders, process_args


# ----------------------------------------------------------------------------- #
#  Seeding / device / forget-class resolution  (same as the original harness)    #
# ----------------------------------------------------------------------------- #
def setup(dataset, forget_class, clustering_type, pipeline, SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Cluster tables are only used to resolve a "head"/"mid"/"tail" keyword into a
    # concrete forget-class index, identical across manual/mean/mvm, so SalUn forgets
    # exactly the SAME class as the KD pipelines and DEEPU.
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
    print("Pipeline     ", pipeline)
    return device, forget_class


# --------------------------------------------------------------------------- #
#  Model builder  (same architecture as load_models, trainable)               #
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
    for p in model.parameters():
        p.requires_grad = True
    print("\nOriginal (teacher) model theta_o loaded")
    return model


def get_targets(dataset_obj):
    if hasattr(dataset_obj, "targets") and dataset_obj.targets is not None:
        return np.asarray(dataset_obj.targets)
    if hasattr(dataset_obj, "samples"):          # ImageFolder fallback
        return np.asarray([s[1] for s in dataset_obj.samples])
    return np.asarray([int(dataset_obj[i][1]) for i in range(len(dataset_obj))])


# --------------------------------------------------------------------------- #
#  Weight saliency mask  (Eq. 3 + Eq. 5)                                        #
# --------------------------------------------------------------------------- #
def compute_saliency_mask(model, forget_loader, device, sparsity, grad_batches=None):
    """m_S = 1(|grad_theta CE_{D_f}| >= gamma), gamma = sparsity-percentile over all weights."""
    model.eval()
    model.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total = 0
    for i, (x, y) in enumerate(forget_loader):
        x, y = x.to(device), y.to(device)
        criterion(model(x), y).backward()
        total += x.size(0)
        if grad_batches is not None and (i + 1) >= grad_batches:
            break
    total = max(total, 1)

    abs_grads = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            abs_grads[n] = (p.grad.detach().abs() / total)
    model.zero_grad(set_to_none=True)

    # Global threshold gamma over the whole gradient vector (paper: median == 50%).
    flat = torch.cat([g.reshape(-1) for g in abs_grads.values()])
    gamma = float(np.percentile(flat.cpu().numpy(), 100.0 * sparsity))

    masks = {}
    salient, tot = 0, 0
    for n, p in model.named_parameters():
        m = (abs_grads[n] >= gamma).float() if n in abs_grads else torch.zeros_like(p)
        masks[n] = m
        salient += int(m.sum().item())
        tot += m.numel()

    print(f"Saliency: gamma={gamma:.3e}  salient weights = {salient}/{tot} "
          f"({100.0*salient/max(tot,1):.1f}% updated, {100.0*(1-salient/max(tot,1)):.1f}% frozen)")
    return masks


# --------------------------------------------------------------------------- #
#  Relabeled forget set + retain set  (D' = D_f' U D_r)                         #
# --------------------------------------------------------------------------- #
class SalUnDataset(Dataset):
    """Yields (image, label, is_forget). Forget samples carry fixed random labels."""
    def __init__(self, base, entries):
        self.base = base
        self.entries = entries          # list of (global_idx, label, is_forget)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        idx, label, is_forget = self.entries[i]
        img, _ = self.base[idx]
        return img, int(label), int(is_forget)


def build_salun_loader(train_loader, forget_class, num_classes, batch_size, seed):
    ds = train_loader.dataset
    targets = get_targets(ds)
    forget_idx = np.where(targets == forget_class)[0]
    retain_idx = np.where(targets != forget_class)[0]

    rng = np.random.RandomState(seed)
    entries = []
    # Relabel forgetting set ONCE with random labels y' != y  (Algorithm 1, D_f').
    for idx in forget_idx:
        y_true = int(targets[idx])
        y_rand = rng.randint(0, num_classes - 1)
        if y_rand >= y_true:             # map uniformly onto {0..K-1}\{y_true}
            y_rand += 1
        entries.append((int(idx), int(y_rand), 1))
    for idx in retain_idx:
        entries.append((int(idx), int(targets[idx]), 0))

    print(f"|D_f'| = {len(forget_idx)} (random-relabeled)   |D_r| = {len(retain_idx)}")
    loader = DataLoader(SalUnDataset(ds, entries), batch_size=batch_size,
                        shuffle=True, num_workers=0)
    return loader


def forget_only_loader(train_loader, forget_class, batch_size):
    """Loader over just the forget-class training images, for the saliency gradient."""
    ds = train_loader.dataset
    targets = get_targets(ds)
    forget_idx = np.where(targets == forget_class)[0].tolist()
    return DataLoader(Subset(ds, forget_idx), batch_size=batch_size, shuffle=False, num_workers=0)


# --------------------------------------------------------------------------- #
#  Saliency-masked RL fine-tuning  (Eq. 6 + Algorithm 1)                        #
# --------------------------------------------------------------------------- #
def salun_unlearn(model, salun_loader, masks, device, epochs, lr, alpha):
    # Plain SGD (no momentum / weight decay) so masked grads == frozen non-salient weights.
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    ce = nn.CrossEntropyLoss()
    mask_list = [(p, masks[n]) for n, p in model.named_parameters() if n in masks]

    model.train()
    for epoch in range(epochs):
        running = 0.0
        for imgs, labels, is_forget in salun_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            is_forget = is_forget.to(device).bool()

            logits = model(imgs)
            f_mask, r_mask = is_forget, ~is_forget

            # Eq. (6): forget-term (coeff 1) + alpha * retain-term, each a mean over its subset.
            loss = torch.zeros((), device=device)
            if f_mask.any():
                loss = loss + ce(logits[f_mask], labels[f_mask])
            if r_mask.any():
                loss = loss + alpha * ce(logits[r_mask], labels[r_mask])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Mask the gradient: theta <- theta - eta (m_S ⊙ g)
            with torch.no_grad():
                for p, m in mask_list:
                    if p.grad is not None:
                        p.grad.mul_(m)
            optimizer.step()
            running += float(loss.item())

        print(f"Epoch [{epoch+1}/{epochs}]  mean batch loss = {running/max(len(salun_loader),1):.4f}")
    return model


# --------------------------------------------------------------------------- #
#  Evaluation  (identical metric to the original train() eval block / DEEPU)   #
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
    p = argparse.ArgumentParser(description="SalUn class-level machine unlearning")
    # --- shared with the original harness / DEEPU ---
    p.add_argument("--imb-factor", default="200")
    p.add_argument("--dataset", default="cifar10", help="cifar10 | cifar100 | food101 | places365")
    p.add_argument("--forget-class", default="head", help="head | mid | tail | numeric")
    p.add_argument("--clustering-type", default="manual", help="only used to resolve the forget-class index")
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # --- SalUn hyperparameters ---
    p.add_argument("--salun-sparsity", type=float, default=0.5,
                   help="fraction of weights left UNCHANGED (0.5 = median threshold, paper default)")
    p.add_argument("--salun-epochs", type=int, default=10, help="RL fine-tuning epochs (paper: 10)")
    p.add_argument("--salun-lr", type=float, default=0.01, help="SGD learning rate eta")
    p.add_argument("--salun-alpha", type=float, default=1.0, help="retain-set regularisation weight alpha")
    p.add_argument("--salun-grad-batches", type=int, default=None,
                   help="cap batches for the saliency gradient (None = full pass over D_f)")
    return p.parse_known_args()   # tolerate leftover --deepu-* flags from shared config lines


def salun_save_path(student_save_dir, imb_factor, forget_class):
    for tok in ("nor_im", "imp_im", "orcl_im", "deepu_im"):
        if tok in student_save_dir:
            return student_save_dir.replace(tok, "salun_im", 1)
    d = os.path.dirname(student_save_dir)
    return os.path.join(d, f"salun_im{imb_factor}_cls{forget_class}.pth")


def main():
    args, unknown = parse_args()
    if unknown:
        print(f"[note] ignoring unrelated args: {unknown}")

    batch_size = 32

    device, forget_class = setup(args.dataset, args.forget_class,
                                 args.clustering_type, "salun", args.seed)

    (TRAIN_DATA, TEST_DATA, num_classes, num_epochs, TEACHER_PATH,
     STUDENT_SAVE_DIR, ORACLE_SAVE_PATH, P_AVG_PLOT_DIR, P_K_BAR_PLOT_DIR) = \
        process_args(args.dataset, args.imb_factor, args.forget_class,
                     args.clustering_type, "nor")

    train_loader, test_loader = data_loaders(
        args.dataset, TRAIN_DATA, TEST_DATA, batch_size, "nor", forget_class)

    model = build_model(args.dataset, device, num_classes, TEACHER_PATH)

    # --- SalUn ---
    t0 = time.perf_counter()
    masks = compute_saliency_mask(model, forget_only_loader(train_loader, forget_class, batch_size),
                                  device, args.salun_sparsity, args.salun_grad_batches)
    salun_loader = build_salun_loader(train_loader, forget_class, num_classes, batch_size, args.seed)
    model = salun_unlearn(model, salun_loader, masks, device,
                          args.salun_epochs, args.salun_lr, args.salun_alpha)
    dt = time.perf_counter() - t0
    print(f"\nSalUn unlearning wall-clock: {dt:.4f} s")

    # --- identical evaluation ---
    print("\n==== Post-unlearning evaluation ====")
    forget_acc, retain_acc = evaluate(model, test_loader, device, num_classes, forget_class)
    print(f"Unlearning Accuracy (UA = 100 - ACC_f): {100.0 - forget_acc:.2f}%")

    # --- save alongside the other methods ---
    save_path = salun_save_path(STUDENT_SAVE_DIR, args.imb_factor, forget_class)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved unlearned model to: {save_path}")


if __name__ == "__main__":
    main()