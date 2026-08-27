"""
UL (Uncertainty Learning) Unlearning  -- unimodal, harness-matched
==================================================================
Push the model toward maximum uncertainty on forget samples via KL divergence to
the uniform distribution. Fine-tunes on the FORGET SET ONLY (no retain data, no
true labels). Core objective identical to the original ADVANCE UL script; only the
interface is adapted to the single-head long-tailed harness used by the other
baselines (DEEPU / SalUn / DELETE / L-CODEC).

CORE UNLEARNING LOGIC (unchanged)
---------------------------------
  uncertainty_loss(logits) = KL(uniform || softmax(logits))
      uniform  = full_like(softmax(logits), 1/num_classes).detach()
      loss     = KLDivLoss(reduction='batchmean')(log_softmax(logits), uniform)
  Training: forget set only, Adam, CosineAnnealingLR(T_max=epochs), early stopping
  on best retain accuracy with patience; best checkpoint reloaded at the end.
  (The original summed this loss over three logits heads; a single-head model has
   one output, so the same objective is applied to that one head.)

HELD IDENTICAL TO THE OTHER BASELINES
-------------------------------------
  Architecture (resnet18/50 + CIFAR conv1/maxpool tweak + fc->num_classes), teacher
  checkpoint theta_o, loaders, forget-class resolution, seeding, evaluation protocol,
  and the shared results CSV.

USAGE (reuses your config matrix; extra flags from other methods are ignored)
-----------------------------------------------------------------------------
  python ul.py --dataset cifar10 --imb-factor 100 --forget-class head \
               --clustering-type manual --seed 18 \
               --ul-lr 1e-5 --ul-epochs 15 --ul-patience 3
"""

import os
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
#  CORE UL LOSS  (unchanged from the original script)                            #
# ----------------------------------------------------------------------------- #
def uncertainty_loss(logits, num_classes):
    """KL divergence between model output and the uniform distribution."""
    uniform = torch.full_like(
        torch.softmax(logits, dim=1),
        fill_value=1.0 / num_classes
    ).detach()
    log_probs = torch.log_softmax(logits, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(log_probs, uniform)
    return loss


def train_one_epoch_UL(model, forget_loader, optimizer, device, epoch, num_classes):
    """UL training epoch. Only forget set is used. No retain set, no true labels."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_max_prob = 0.0

    for images, _labels in forget_loader:      # labels intentionally unused
        images = images.to(device)
        optimizer.zero_grad()

        logits = model(images)
        loss = uncertainty_loss(logits, num_classes)   # single head (same objective)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        with torch.no_grad():
            max_prob = torch.softmax(logits, dim=1).max(dim=1).values.mean().item()
            total_max_prob += max_prob * images.size(0)

    n = max(total_samples, 1)
    avg_max_prob = total_max_prob / n
    perfect = 1.0 / num_classes
    print(f"\n  Epoch {epoch} Train Summary:")
    print(f"    Uncertainty Loss     : {total_loss/n:.4f}")
    print(f"    Avg Max Prob         : {avg_max_prob:.4f}")
    print(f"    Perfect Uncertainty  : {perfect:.4f}")
    print(f"    Gap from perfect     : {abs(avg_max_prob - perfect):.4f}")


# ----------------------------------------------------------------------------- #
#  Harness helpers  (identical to the other baselines)                           #
# ----------------------------------------------------------------------------- #
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
    print(f"|D_f| = {len(forget_idx)}   (UL trains on the forget set only; no D_r, no labels)")
    return DataLoader(Subset(ds, forget_idx), batch_size=batch_size, shuffle=True, num_workers=0)


@torch.no_grad()
def evaluate(model, test_loader, device, num_classes, forget_class, verbose=True):
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
        if verbose:
            print(f"{i}: {acc:.2f}%")
        if i == forget_class:
            forget_acc = acc
        else:
            retain_acc += acc; retain_classes += 1
    retain_acc /= max(retain_classes, 1)
    if verbose:
        print(f"\nForget-class accuracy (ACC_f, lower=better): {forget_acc:.2f}%")
        print(f"Retain Accuracy (mean over non-forget classes): {retain_acc:.2f}%")
    return forget_acc, retain_acc


# Shared, method-agnostic results schema (same as the other baselines).
CSV_FIELDS = [
    "timestamp", "method", "dataset", "imb_factor", "forget_class",
    "clustering_type", "seed", "forget_acc", "retain_acc", "ua",
    "wall_time_s", "hyperparams", "save_path",
]


def append_result_csv(csv_path, row):
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


# ----------------------------------------------------------------------------- #
#  UL run  (training loop / early-stopping logic preserved from the original)    #
# ----------------------------------------------------------------------------- #
def run_UL(model, forget_loader, test_loader, device, num_classes, forget_class,
           save_path, lr, epochs, patience):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_retain_acc = -1.0
    best_epoch = 0
    patience_count = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        print(f"\n{'-'*60}\nEpoch [{epoch}/{epochs}]\n{'-'*60}")
        train_one_epoch_UL(model, forget_loader, optimizer, device, epoch, num_classes)
        scheduler.step()

        # early-stop selection on retain accuracy (quiet eval each epoch)
        _fa, retain_acc = evaluate(model, test_loader, device, num_classes, forget_class, verbose=False)
        print(f"  [epoch {epoch}] retain acc = {retain_acc:.2f}%")

        if retain_acc > best_retain_acc:
            best_retain_acc = retain_acc
            best_epoch = epoch
            patience_count = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  New best (epoch {epoch}, retain={retain_acc:.2f}%)")
        else:
            patience_count += 1
            print(f"  No improvement. Patience {patience_count}/{patience}")
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # reload best checkpoint
    model.load_state_dict(best_state)
    print(f"\nRestored best epoch {best_epoch} (retain={best_retain_acc:.2f}%)")
    return model, best_epoch


# ----------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="UL (uncertainty learning) class-level machine unlearning")
    p.add_argument("--imb-factor", default="200")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--forget-class", default="head", help="head | mid | tail | numeric")
    p.add_argument("--clustering-type", default="manual", help="only used to resolve the forget-class index")
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # UL hyperparameters (defaults match the original script's __main__ call)
    p.add_argument("--ul-lr", type=float, default=1e-5, help="Adam learning rate")
    p.add_argument("--ul-epochs", type=int, default=15, help="max UL epochs")
    p.add_argument("--ul-patience", type=int, default=3, help="early-stopping patience on retain acc")
    p.add_argument("--ul-batch-size", type=int, default=32)
    p.add_argument("--csv", default="/export/home/achyut/Simarjeet/results/ul.csv",
                   help="append a results row to this CSV (shared schema; '' disables)")
    return p.parse_known_args()


def ul_save_path(student_save_dir, imb_factor, forget_class):
    for tok in ("nor_im", "imp_im", "orcl_im", "deepu_im", "salun_im", "delete_im", "lcodec_im"):
        if tok in student_save_dir:
            return student_save_dir.replace(tok, "ul_im", 1)
    d = os.path.dirname(student_save_dir)
    return os.path.join(d, f"ul_im{imb_factor}_cls{forget_class}.pth")


def main():
    args, unknown = parse_args()
    if unknown:
        print(f"[note] ignoring unrelated args: {unknown}")

    batch_size = args.ul_batch_size
    device, forget_class = setup(args.dataset, args.forget_class,
                                 args.clustering_type, "ul", args.seed)

    (TRAIN_DATA, TEST_DATA, num_classes, num_epochs, TEACHER_PATH,
     STUDENT_SAVE_DIR, ORACLE_SAVE_PATH, P_AVG_PLOT_DIR, P_K_BAR_PLOT_DIR) = \
        process_args(args.dataset, args.imb_factor, args.forget_class, args.clustering_type, "nor")

    train_loader, test_loader = data_loaders(
        args.dataset, TRAIN_DATA, TEST_DATA, batch_size, "nor", forget_class)

    model = build_model(args.dataset, device, num_classes, TEACHER_PATH)
    forget_loader = forget_loader_from_train(train_loader, forget_class, batch_size)

    save_path = ul_save_path(STUDENT_SAVE_DIR, args.imb_factor, forget_class)

    # --- UL ---
    t0 = time.perf_counter()
    model, best_epoch = run_UL(model, forget_loader, test_loader, device,
                               num_classes, forget_class, save_path,
                               lr=args.ul_lr, epochs=args.ul_epochs, patience=args.ul_patience)
    dt = time.perf_counter() - t0
    print(f"\nUL unlearning wall-clock: {dt:.4f} s")

    print("\n==== Post-unlearning evaluation ====")
    forget_acc, retain_acc = evaluate(model, test_loader, device, num_classes, forget_class)
    print(f"Unlearning Accuracy (UA = 100 - ACC_f): {100.0 - forget_acc:.2f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved unlearned model to: {save_path}")

    # --- append a row to the shared results CSV ---
    import datetime
    hyper = (f"lr={args.ul_lr};epochs={args.ul_epochs};patience={args.ul_patience};"
             f"batch_size={batch_size};best_epoch={best_epoch}")
    append_result_csv(args.csv, {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "method": "ul",
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