"""
SCRUB: SCalable Remembering and Unlearning unBound
(Kurmanji, Triantafillou, Hayes, Triantafillou — NeurIPS 2023,
 "Towards Unbounded Machine Unlearning", arXiv:2302.09880)

Self-contained, faithful reimplementation of SCRUB (Algorithms 1-3) + optional
SCRUB+R rewind (Sec 3.2), wired into the SAME harness as the user's KD script and
the DEEPU / SalUn / DELETE / UL baselines, so methods differ ONLY in the mechanism.

CORE METHOD (unchanged, per the paper's pseudocode)
---------------------------------------------------
  teacher w^o = frozen original; student w^u initialised to w^o.
  d(x) = KL( softmax(f(x; w^o)) || softmax(f(x; w^u)) )        (teacher||student)

  DO-MAX-EPOCH (Alg 2): one epoch over D_f, ASCEND d  ->  push student away from
      teacher on forget:   w <- w + eps * grad d(x_f)     (i.e. minimise -d)
  DO-MIN-EPOCH (Alg 3): one epoch over D_r, DESCEND  alpha*d(x_r) + gamma*CE(x_r,y_r):
      w <- w - eps * grad [ alpha*d(x_r) + gamma*CE ]      (stay close + task loss)

  SCHEDULE (Alg 1): for step i in range(total):
      if i < max_steps:  DO-MAX-EPOCH
      DO-MIN-EPOCH                                          (min-steps continue after
                                                            max-steps stop -> restore retain)
  Optimizer: SGD/Adam, lr 5e-4 decayed x0.1 (paper); alpha, gamma are hyperparameters.

SCRUB+R (optional --scrub-rewind): checkpoint every epoch; build a same-class
  validation set (held-out test images of the forget class); rewind to the checkpoint
  whose forget-set error is closest to that validation error. Useful for the MIA/privacy
  column; for the forgetting table you typically want plain SCRUB (max forget error).

HELD IDENTICAL TO THE OTHER BASELINES
-------------------------------------
  Architecture, teacher checkpoint theta_o, loaders, forget-class resolution, seeding,
  evaluation protocol, and the shared results CSV.

USAGE (reuses your config matrix; extra flags from other methods are ignored)
-----------------------------------------------------------------------------
  python scrub.py --dataset cifar10 --imb-factor 100 --forget-class head \
                  --clustering-type manual --seed 18 \
                  --scrub-max-steps 3 --scrub-min-steps 4 --scrub-alpha 1.0 --scrub-gamma 1.0
"""

import os
import copy
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models

from utils import data_loaders, process_args


# ----------------------------------------------------------------------------- #
#  CORE SCRUB losses  (d = KL(teacher || student))                               #
# ----------------------------------------------------------------------------- #
def kl_teacher_student(student_logits, teacher_logits):
    """d(x) = KL( softmax(teacher) || softmax(student) ), averaged over the batch."""
    p_teacher = F.softmax(teacher_logits, dim=1)
    log_p_student = F.log_softmax(student_logits, dim=1)
    # KLDivLoss(input=log q, target=p) = sum p (log p - log q) = KL(p||q), p=teacher
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean")


def do_max_epoch(student, teacher, forget_loader, optimizer, device):
    """Algorithm 2: ascend d on D_f  =>  minimise -d (push student away from teacher)."""
    student.train()
    for images, _labels in forget_loader:
        images = images.to(device)
        with torch.no_grad():
            t_logits = teacher(images)
        s_logits = student(images)
        loss = -kl_teacher_student(s_logits, t_logits)   # ascend d
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def do_min_epoch(student, teacher, retain_loader, optimizer, device, alpha, gamma):
    """Algorithm 3: descend alpha*d + gamma*CE on D_r (stay close + task loss)."""
    student.train()
    ce = nn.CrossEntropyLoss()
    for images, labels in retain_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            t_logits = teacher(images)
        s_logits = student(images)
        loss = alpha * kl_teacher_student(s_logits, t_logits) + gamma * ce(s_logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


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


def forget_retain_loaders(train_loader, forget_class, forget_bs, retain_bs):
    ds = train_loader.dataset
    targets = get_targets(ds)
    forget_idx = np.where(targets == forget_class)[0].tolist()
    retain_idx = np.where(targets != forget_class)[0].tolist()
    print(f"|D_f| = {len(forget_idx)}   |D_r| = {len(retain_idx)}")
    f_loader = DataLoader(Subset(ds, forget_idx), batch_size=forget_bs, shuffle=True, num_workers=0)
    r_loader = DataLoader(Subset(ds, retain_idx), batch_size=retain_bs, shuffle=True, num_workers=0)
    return f_loader, r_loader


def forget_test_loader(test_loader, forget_class, batch_size):
    """Same-class held-out set for SCRUB+R rewind reference (validation error)."""
    ds = test_loader.dataset
    targets = get_targets(ds)
    idx = np.where(targets == forget_class)[0].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def error_on_loader(model, loader, device):
    """Top-1 error (1 - accuracy) on a loader."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 1.0 - correct / max(total, 1)


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
#  SCRUB run  (Algorithm 1 schedule + optional +R rewind)                        #
# ----------------------------------------------------------------------------- #
def run_scrub(student, teacher, forget_loader, retain_loader, device,
              max_steps, min_steps, alpha, gamma, lr, momentum, weight_decay,
              decay_after, optimizer_name, rewind, fval_loader, fforget_loader):
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if optimizer_name == "adam":
        optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_after, gamma=0.1)

    total_steps = max(max_steps, min_steps)
    checkpoints = []   # (epoch_idx, forget_err_on_Df, state_dict) for rewind

    for i in range(total_steps):
        if i < max_steps:
            do_max_epoch(student, teacher, forget_loader, optimizer, device)   # Alg 2
        do_min_epoch(student, teacher, retain_loader, optimizer, device, alpha, gamma)  # Alg 3
        scheduler.step()

        f_err = error_on_loader(student, fforget_loader, device)
        r_err = error_on_loader(student, retain_loader, device)
        print(f"  [step {i+1}/{total_steps}]  forget_err={f_err:.3f}  retain_err={r_err:.3f}")
        if rewind:
            checkpoints.append((i, f_err,
                                {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}))

    if rewind and checkpoints:
        # SCRUB+R: reference = error on same-class held-out validation set at the last checkpoint
        ref_err = error_on_loader(student, fval_loader, device)
        best = min(checkpoints, key=lambda c: abs(c[1] - ref_err))
        student.load_state_dict(best[2])
        print(f"\nSCRUB+R: validation-ref forget err={ref_err:.3f}; "
              f"rewound to step {best[0]+1} (forget_err={best[1]:.3f})")
    return student


# ----------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="SCRUB class-level machine unlearning")
    p.add_argument("--imb-factor", default="200")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--forget-class", default="head", help="head | mid | tail | numeric")
    p.add_argument("--clustering-type", default="manual", help="only used to resolve the forget-class index")
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # SCRUB hyperparameters (defaults follow the paper's class-unlearning setup)
    p.add_argument("--scrub-max-steps", type=int, default=3, help="# max-epochs (Alg 1); paper class: 2-3")
    p.add_argument("--scrub-min-steps", type=int, default=4, help="# min-epochs (Alg 1); paper class: 3-4")
    p.add_argument("--scrub-alpha", type=float, default=1.0, help="weight on KL(teacher||student) in min-step")
    p.add_argument("--scrub-gamma", type=float, default=1.0, help="weight on CE task loss in min-step")
    p.add_argument("--scrub-lr", type=float, default=5e-4, help="initial lr (paper: 5e-4)")
    p.add_argument("--scrub-momentum", type=float, default=0.9)
    p.add_argument("--scrub-weight-decay", type=float, default=5e-4, help="paper: 5e-4 (large-scale)")
    p.add_argument("--scrub-decay-after", type=int, default=3, help="StepLR: decay lr x0.1 after this many steps")
    p.add_argument("--scrub-optimizer", default="sgd", choices=["sgd", "adam"])
    p.add_argument("--scrub-forget-bs", type=int, default=512, help="forget-set batch size (paper class: 512)")
    p.add_argument("--scrub-retain-bs", type=int, default=128, help="retain-set batch size (paper class: 128)")
    p.add_argument("--scrub-rewind", action="store_true",
                   help="enable SCRUB+R rewind (for the MIA/privacy case; keeps forget err near retrain level)")
    p.add_argument("--csv", default="unlearning_results.csv",
                   help="append a results row to this CSV (shared schema; '' disables)")
    return p.parse_known_args()


def scrub_save_path(student_save_dir, imb_factor, forget_class):
    for tok in ("nor_im", "imp_im", "orcl_im", "deepu_im", "salun_im", "delete_im", "lcodec_im", "ul_im"):
        if tok in student_save_dir:
            return student_save_dir.replace(tok, "scrub_im", 1)
    d = os.path.dirname(student_save_dir)
    return os.path.join(d, f"scrub_im{imb_factor}_cls{forget_class}.pth")


def main():
    args, unknown = parse_args()
    if unknown:
        print(f"[note] ignoring unrelated args: {unknown}")

    device, forget_class = setup(args.dataset, args.forget_class,
                                 args.clustering_type, "scrub", args.seed)

    (TRAIN_DATA, TEST_DATA, num_classes, num_epochs, TEACHER_PATH,
     STUDENT_SAVE_DIR, ORACLE_SAVE_PATH, P_AVG_PLOT_DIR, P_K_BAR_PLOT_DIR) = \
        process_args(args.dataset, args.imb_factor, args.forget_class, args.clustering_type, "nor")

    # loader batch size here only affects the test loader; SCRUB uses its own D_f/D_r bs
    train_loader, test_loader = data_loaders(
        args.dataset, TRAIN_DATA, TEST_DATA, args.scrub_retain_bs, "nor", forget_class)

    teacher = build_model(args.dataset, device, num_classes, TEACHER_PATH)
    student = copy.deepcopy(teacher)                       # w^u <- w^o
    for p in student.parameters():
        p.requires_grad_(True)

    forget_loader, retain_loader = forget_retain_loaders(
        train_loader, forget_class, args.scrub_forget_bs, args.scrub_retain_bs)
    fforget_loader = DataLoader(forget_loader.dataset, batch_size=args.scrub_retain_bs,
                                shuffle=False, num_workers=0)          # for error tracking
    fval_loader = forget_test_loader(test_loader, forget_class, args.scrub_retain_bs)  # +R reference

    # --- SCRUB ---
    t0 = time.perf_counter()
    student = run_scrub(
        student, teacher, forget_loader, retain_loader, device,
        max_steps=args.scrub_max_steps, min_steps=args.scrub_min_steps,
        alpha=args.scrub_alpha, gamma=args.scrub_gamma,
        lr=args.scrub_lr, momentum=args.scrub_momentum, weight_decay=args.scrub_weight_decay,
        decay_after=args.scrub_decay_after, optimizer_name=args.scrub_optimizer,
        rewind=args.scrub_rewind, fval_loader=fval_loader, fforget_loader=fforget_loader,
    )
    dt = time.perf_counter() - t0
    print(f"\nSCRUB unlearning wall-clock: {dt:.4f} s")

    print("\n==== Post-unlearning evaluation ====")
    forget_acc, retain_acc = evaluate(student, test_loader, device, num_classes, forget_class)
    print(f"Unlearning Accuracy (UA = 100 - ACC_f): {100.0 - forget_acc:.2f}%")

    save_path = scrub_save_path(STUDENT_SAVE_DIR, args.imb_factor, forget_class)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(student.state_dict(), save_path)
    print(f"\nSaved unlearned model to: {save_path}")

    import datetime
    hyper = (f"max_steps={args.scrub_max_steps};min_steps={args.scrub_min_steps};"
             f"alpha={args.scrub_alpha};gamma={args.scrub_gamma};lr={args.scrub_lr};"
             f"opt={args.scrub_optimizer};forget_bs={args.scrub_forget_bs};"
             f"retain_bs={args.scrub_retain_bs};rewind={args.scrub_rewind}")
    append_result_csv(args.csv, {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "method": "scrub_r" if args.scrub_rewind else "scrub",
        "dataset": args.dataset, "imb_factor": args.imb_factor,
        "forget_class": forget_class, "clustering_type": args.clustering_type,
        "seed": args.seed,
        "forget_acc": f"{forget_acc:.2f}", "retain_acc": f"{retain_acc:.2f}",
        "ua": f"{100.0 - forget_acc:.2f}", "wall_time_s": f"{dt:.2f}",
        "hyperparams": hyper, "save_path": save_path,
    })


if __name__ == "__main__":
    main()