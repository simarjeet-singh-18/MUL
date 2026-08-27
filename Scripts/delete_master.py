"""
DELETE: DEcoupLEd Distillation To Erase
(Zhou, Zheng, Mo, Lu, Lin, Zheng — CVPR 2025,
 "Decoupled Distillation to Erase: A General Unlearning Method for Any Class-centric Tasks")

Self-contained, faithful reimplementation of DELETE's core method (Algorithm 1,
Eq. (5) + Appendix H reformulation), wired into the SAME experimental harness as
the user's KD-based unlearning script and the DEEPU / SalUn baselines, so all
methods differ ONLY in the unlearning mechanism. No dependency on the other scripts.

CORE METHOD (exactly as in the paper)
-------------------------------------
DELETE is a post-hoc mask-distillation method that uses ONLY the forget set D_f
(no remaining data, no pre-training intervention). It freezes a copy of the
original model theta_o and, for each forget-class image x (true class u):

    z            = f_{theta_o}(x)                       # frozen logits
    Mask'_u(z)   = z with the u-th (true-label) logit set to -inf
    p (target)   = Softmax( Mask'_u(z / T) )            # forget prob 0; others
                                                        # proportional to original -> retention
    q            = Softmax( f_theta(x) )                # unlearn-model output
    L            = KL( p || q )                         # Eq. (5)

The masked target simultaneously enforces the *forgetting condition* (p_u = 0)
and the *retention condition* (p_i proportional to Softmax(z/T)_i for i != u,
i.e. the original model's "dark knowledge" over the remaining classes). T = 1
is the paper's default retention setting.

Implemented with nn.KLDivLoss: input = log_softmax(student), target = p, so it
computes sum p (log p - log q) = KL(p||q) averaged per sample (batchmean).

HYPERPARAMETERS (paper appendix, Tab. S6 + Sec. I.2)
----------------------------------------------------
  Optimizer         : SGD, momentum 0.9, weight decay 5e-4
  LR scheduler      : StepLR(step_size=40, gamma=0.1)   (inert within 20 epochs)
  Unlearning epochs : 20   ("training-based unlearning methods ... 20 epochs")
  LR                : searched in [1e-7, 1e-2] in the paper -> exposed as a flag
  Temperature T     : 1.0 (default retention condition)

HELD IDENTICAL TO THE ORIGINAL SCRIPT / DEEPU / SalUn (so only the method differs)
---------------------------------------------------------------------------------
  * Architecture (resnet18/50 + CIFAR conv1/maxpool tweak + fc->num_classes),
    teacher checkpoint (theta_o), loaders, forget-class resolution, seeding, and
    the evaluation protocol (per-class acc + Retain Accuracy over non-forget classes)
    are the SAME as the other baselines and consistent with the KD harness.
  * batch_size default 32 to match the harness (paper used 128; --delete-batch-size).

USAGE (reuses your existing config matrix; extra --deepu-*/--salun-* flags ignored)
-----------------------------------------------------------------------------------
  python delete.py --dataset cifar10 --imb-factor 100 --forget-class head \
                   --clustering-type manual --seed 18 \
                   --delete-epochs 20 --delete-lr 0.01 --delete-T 1.0
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


def forget_loader_from_train(train_loader, forget_class, batch_size):
    """Loader over ONLY the forget-class training images (DELETE uses D_f alone)."""
    ds = train_loader.dataset
    targets = get_targets(ds)
    forget_idx = np.where(targets == forget_class)[0].tolist()
    print(f"|D_f| = {len(forget_idx)}   (DELETE trains on the forget set only; no D_r)")
    return DataLoader(Subset(ds, forget_idx), batch_size=batch_size, shuffle=True, num_workers=0)


# --------------------------------------------------------------------------- #
#  DELETE mask-distillation unlearning  (Algorithm 1, Eq. 5)                    #
# --------------------------------------------------------------------------- #
def _set_bn_eval(module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def delete_unlearn(model, forget_loader, device, epochs, lr, momentum,
                   weight_decay, T, step_size, gamma, freeze_bn):
    # Frozen copy of theta_o provides the distillation target.
    frozen = copy.deepcopy(model)
    for p in frozen.parameters():
        p.requires_grad_(False)
    frozen.eval()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    kl = nn.KLDivLoss(reduction="batchmean")

    model.train()
    if freeze_bn:
        _set_bn_eval(model)   # optional: keep BN stats fixed (helps single-class LT retention)

    for epoch in range(epochs):
        running, nb = 0.0, 0
        for images, labels in forget_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                z = frozen(images)                                  # frozen logits
                mask = torch.zeros_like(z)
                mask[torch.arange(z.size(0)), labels] = float("-inf")   # Mask'_u : -inf at true class u
                soft_targets = torch.softmax(mask + z / T, dim=1)       # p = Softmax(Mask'_u(z/T))

            logits = model(images)                                  # unlearn-model logits
            # L = KL(p || q); KLDivLoss(input=log q, target=p) = sum p (log p - log q)
            loss = kl(torch.log_softmax(logits, dim=1), soft_targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item()); nb += 1

        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}]  mean batch KL = {running/max(nb,1):.5f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
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
    p = argparse.ArgumentParser(description="DELETE class-level machine unlearning")
    # --- shared with the original harness / DEEPU / SalUn ---
    p.add_argument("--imb-factor", default="200")
    p.add_argument("--dataset", default="cifar10", help="cifar10 | cifar100 | food101 | places365")
    p.add_argument("--forget-class", default="head", help="head | mid | tail | numeric")
    p.add_argument("--clustering-type", default="manual", help="only used to resolve the forget-class index")
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # --- DELETE hyperparameters (paper appendix) ---
    p.add_argument("--delete-epochs", type=int, default=20, help="unlearning epochs (paper: 20)")
    p.add_argument("--delete-lr", type=float, default=0.01, help="SGD lr (paper searched [1e-7, 1e-2])")
    p.add_argument("--delete-momentum", type=float, default=0.9, help="SGD momentum (paper: 0.9)")
    p.add_argument("--delete-weight-decay", type=float, default=5e-4, help="weight decay (paper: 5e-4)")
    p.add_argument("--delete-T", type=float, default=1.0, help="retention temperature T (paper default: 1.0)")
    p.add_argument("--delete-step-size", type=int, default=40, help="StepLR step size (paper: 40)")
    p.add_argument("--delete-gamma", type=float, default=0.1, help="StepLR gamma (paper: 0.1)")
    p.add_argument("--delete-batch-size", type=int, default=32,
                   help="batch size (harness default 32; paper used 128)")
    p.add_argument("--delete-freeze-bn", action="store_true",
                   help="freeze BN running stats during unlearning (optional; can help single-class LT retention)")
    return p.parse_known_args()   # tolerate leftover --deepu-*/--salun-* flags from shared config lines


def delete_save_path(student_save_dir, imb_factor, forget_class):
    for tok in ("nor_im", "imp_im", "orcl_im", "deepu_im", "salun_im"):
        if tok in student_save_dir:
            return student_save_dir.replace(tok, "delete_im", 1)
    d = os.path.dirname(student_save_dir)
    return os.path.join(d, f"delete_im{imb_factor}_cls{forget_class}.pth")


def main():
    args, unknown = parse_args()
    if unknown:
        print(f"[note] ignoring unrelated args: {unknown}")

    batch_size = args.delete_batch_size

    device, forget_class = setup(args.dataset, args.forget_class,
                                 args.clustering_type, "delete", args.seed)

    (TRAIN_DATA, TEST_DATA, num_classes, num_epochs, TEACHER_PATH,
     STUDENT_SAVE_DIR, ORACLE_SAVE_PATH, P_AVG_PLOT_DIR, P_K_BAR_PLOT_DIR) = \
        process_args(args.dataset, args.imb_factor, args.forget_class,
                     args.clustering_type, "nor")

    train_loader, test_loader = data_loaders(
        args.dataset, TRAIN_DATA, TEST_DATA, batch_size, "nor", forget_class)

    model = build_model(args.dataset, device, num_classes, TEACHER_PATH)
    forget_loader = forget_loader_from_train(train_loader, forget_class, batch_size)

    # --- DELETE ---
    t0 = time.perf_counter()
    model = delete_unlearn(
        model, forget_loader, device,
        epochs=args.delete_epochs, lr=args.delete_lr, momentum=args.delete_momentum,
        weight_decay=args.delete_weight_decay, T=args.delete_T,
        step_size=args.delete_step_size, gamma=args.delete_gamma,
        freeze_bn=args.delete_freeze_bn,
    )
    dt = time.perf_counter() - t0
    print(f"\nDELETE unlearning wall-clock: {dt:.4f} s")

    # --- identical evaluation ---
    print("\n==== Post-unlearning evaluation ====")
    forget_acc, retain_acc = evaluate(model, test_loader, device, num_classes, forget_class)
    print(f"Unlearning Accuracy (UA = 100 - ACC_f): {100.0 - forget_acc:.2f}%")

    # --- save alongside the other methods ---
    save_path = delete_save_path(STUDENT_SAVE_DIR, args.imb_factor, forget_class)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved unlearned model to: {save_path}")


if __name__ == "__main__":
    main()