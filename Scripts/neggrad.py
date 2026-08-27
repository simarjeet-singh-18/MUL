import time
import torch
import random
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from utils import SavedDataset, process_args
from soa_utils import evaluate_split, print_accuracy_matrix, load_model



def get_forget_retain_splits(
    TRAIN_DATA,
    TEST_DATA,
    num_classes,
    forget_class=0,
    imb_factor=200,
    val_ratio=0,
    test_ratio=0.2,
    seed=42,
    dataset="cifar100"
):
    """
    Returns:
        forget_train, forget_val, forget_test
        retain_train, retain_val, retain_test
    """
    
    # Make transforms for particular datasets
    if dataset == "cifar10":
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])

    elif dataset == "cifar100":
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761)
            )
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761)
            )
        ])
        
    elif dataset == "food101":
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)), 
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225])
        ])
        
    elif dataset == "places365":
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std  = [0.229, 0.224, 0.225]
            )
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std  = [0.229, 0.224, 0.225]
            )
        ])
    
    # Load datasets
    
    if dataset == "cifar10" or dataset == "cifar100":

        _dataset = SavedDataset(
            file_path=TRAIN_DATA,
            transform=transform_train
        )
        
    elif dataset == "food101" or dataset == "places365":
        
        _dataset = torchvision.datasets.ImageFolder(
            TRAIN_DATA,
            transform=transform_train
        )
        
    if dataset == "cifar10":
    
        dataset_test = torchvision.datasets.CIFAR10(
            root=TEST_DATA,
            train=False,
            download=True,
            transform=transform_test
        )
    
    elif dataset == "cifar100":

        dataset_test = torchvision.datasets.CIFAR100(
            root=TEST_DATA,
            train=False,
            download=True,
            transform=transform_test
        )
        
    elif dataset == "food101" or dataset == "places365":
        
        dataset_test = torchvision.datasets.ImageFolder(
            root=TEST_DATA,
            transform=transform_test
        )

    targets = np.array(_dataset.targets)
    targets_test = np.array(dataset_test.targets)

    # Forget samples
    forget_idx = np.where(targets == forget_class)[0]
    forget_idx_test = np.where(targets_test == forget_class)[0]

    # Retain samples
    retain_idx = np.where(targets != forget_class)[0]
    retain_idx_test = np.where(targets_test != forget_class)[0]

    rng = np.random.default_rng(seed)

    rng.shuffle(forget_idx)
    rng.shuffle(retain_idx)
    rng.shuffle(forget_idx_test)
    rng.shuffle(retain_idx_test)

    def split_indices(indices):
        n = len(indices)

        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)

        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        return train_idx, val_idx, test_idx

    forget_train_idx, forget_val_idx, forget_test_idx = split_indices(forget_idx)
    retain_train_idx, retain_val_idx, retain_test_idx = split_indices(retain_idx)

    forget_train = Subset(_dataset, forget_train_idx)
    forget_val = Subset(_dataset, forget_val_idx)
    forget_test = Subset(dataset_test, forget_idx_test)

    retain_train = Subset(_dataset, retain_train_idx)
    retain_val = Subset(_dataset, retain_val_idx)
    retain_test = Subset(dataset_test, retain_idx_test)

    return (
        forget_train,
        forget_val,
        forget_test,
        retain_train,
        retain_val,
        retain_test,
        num_classes
    )



def train_one_epoch_neggrad(
    model,
    retain_loader,
    forget_loader,
    optimizer,
    device,
    epoch,
    forget_lambda=1.0
):
    """
    NegGrad training epoch.

    Loss = retain_loss - lambda * forget_loss

    Args:
        forget_lambda: Weight for forget loss
    """

    model.train()

    criterion = nn.CrossEntropyLoss()

    forget_iter = iter(forget_loader)

    total_loss = 0.0
    total_retain_loss = 0.0
    total_forget_loss = 0.0

    total_correct = 0
    total_samples = 0

    for r_imgs, r_labels in tqdm(
        retain_loader,
        desc=f"  Epoch {epoch} [NegGrad]"
    ):

        r_imgs = r_imgs.to(device)
        r_labels = r_labels.to(device)

        try:
            f_imgs, f_labels = next(forget_iter)

        except StopIteration:

            forget_iter = iter(forget_loader)
            f_imgs, f_labels = next(forget_iter)

        f_imgs = f_imgs.to(device)
        f_labels = f_labels.to(device)

        optimizer.zero_grad()

        # ── Retain forward ──

        out_r = model(r_imgs)

        if isinstance(out_r, tuple):
            out_r = out_r[0]

        r_loss = criterion(
            out_r,
            r_labels
        )

        # ── Forget forward ──

        out_f = model(f_imgs)

        if isinstance(out_f, tuple):
            out_f = out_f[0]

        f_loss = criterion(
            out_f,
            f_labels
        )

        # ── NegGrad objective ──

        loss = r_loss - forget_lambda * f_loss

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * r_labels.size(0)
        )

        total_retain_loss += (
            r_loss.item() * r_labels.size(0)
        )

        total_forget_loss += (
            f_loss.item() * f_labels.size(0)
        )

        preds = out_r.argmax(dim=1)

        total_correct += (
            preds == r_labels
        ).sum().item()

        total_samples += r_labels.size(0)

    n = total_samples

    print(f"\n  Epoch {epoch} Train Summary:")
    print(
        f"    Total Loss   : "
        f"{total_loss/n:.4f}"
    )

    print(
        f"    Retain Loss  : "
        f"{total_retain_loss/n:.4f}  "
        f"(minimized)"
    )

    print(
        f"    Forget Loss  : "
        f"{total_forget_loss/len(forget_loader.dataset):.4f}  "
        f"(maximized)"
    )

    print(
        f"    Retain Acc   : "
        f"{100*total_correct/n:.2f}%"
    )



def run_neggrad_unlearning(TRAIN_DATA, TEST_DATA, num_classes, original_ckpt, save_ckpt_path, forget_class, imb_factor, dataset,
                           device, lr=1e-4, epochs=15, batch_size=32,
                           num_workers=4, patience=5, forget_lambda=1.0):
    """
    Run NegGrad unlearning on ADVANCE dataset.
    
    Args:
        original_ckpt: Path to original trained model
        save_ckpt_path: Path to save unlearned model
        forget_class: Class name to forget (e.g., "airport")
        device: torch device
        lr: Learning rate
        epochs: Max epochs
        batch_size: Batch size
        num_workers: DataLoader workers
        patience: Early stopping patience
        forget_lambda: Weight for forget loss
    """
    print("\n" + "="*70)
    print("BUILDING DATASETS")
    print("="*70)
    
    # Get splits
    # forget_train, forget_val, forget_test = get_forget_splits(
    #     forget_class=forget_class, val_ratio=0.2, test_ratio=0.1, seed=42
    # )
    # retain_train, retain_val, retain_test = get_retain_splits(
    #     forget_class=forget_class, val_ratio=0.2, test_ratio=0.1, seed=42
    # )
    
    forget_train, forget_val, forget_test, retain_train, retain_val, retain_test, num_classes = get_forget_retain_splits(TRAIN_DATA, TEST_DATA, num_classes, dataset=dataset, imb_factor=imb_factor, forget_class=forget_class)
    
    print(f"\n  Forget class: {forget_class}")
    print(f"  Forget train: {len(forget_train)}")
    print(f"  Forget val  : {len(forget_val)}")
    print(f"  Forget test : {len(forget_test)}")
    print(f"  Retain train: {len(retain_train)}")
    print(f"  Retain val  : {len(retain_val)}")
    print(f"  Retain test : {len(retain_test)}")
    
    # Create dataloaders
    forget_train_loader = DataLoader(forget_train, batch_size=batch_size, 
                                     shuffle=True, num_workers=num_workers)
    forget_test_loader = DataLoader(forget_test, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    retain_train_loader = DataLoader(retain_train, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers)
    retain_test_loader = DataLoader(retain_test, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    
    # Combined validation set
    from torch.utils.data import ConcatDataset
    val_dataset = ConcatDataset([forget_val, retain_val])
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    
    print("\n" + "="*70)
    print("LOADING ORIGINAL MODEL")
    print("="*70)
    
    # model = torchvision.models.resnet50(weights=None)
    # model.conv1 = nn.Conv2d(
    # 3, 64, kernel_size=3, stride=1, padding=1, bias=False
    # )
    # model.maxpool = nn.Identity()
    # model.fc = nn.Linear(model.fc.in_features, 100)
    model = load_model(dataset, num_classes)
    model.load_state_dict(torch.load(original_ckpt, map_location=device))
    model = model.to(device)

    
    # model = load_model(original_ckpt, NUM_CLASSES, device)
    
    print("\nEvaluating BEFORE unlearning...")
    matrix_before = evaluate_split(model, forget_test_loader, retain_test_loader, device)
    
    print_accuracy_matrix(matrix_before, title="BEFORE UNLEARNING (Test Set)")
    
    full_set = ConcatDataset([retain_test, forget_test])
    full_loader = DataLoader(
        full_set,
        batch_size=32,
        shuffle=False
    )
    
    matrix_full = evaluate_split(model, forget_test_loader, full_loader, device)
    print_accuracy_matrix(matrix_full, title="FULL TEST")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\n" + "="*70)
    print("NegGrad UNLEARNING")
    print(f"forget_lambda = {forget_lambda}")
    print("="*70)
    
    best_retain_acc = 0.0
    best_epoch = 0
    patience_count = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'-'*70}")
        print(f"Epoch [{epoch}/{epochs}]")
        print(f"{'-'*70}")
        
        train_one_epoch_neggrad(model, retain_train_loader, forget_train_loader,
                               optimizer, device, epoch, forget_lambda)
        scheduler.step()
        
        # Validate on retain set only (we want to preserve retain performance)
        matrix_val = evaluate_split(model, forget_test_loader, retain_test_loader, device)
        print_accuracy_matrix(matrix_val, title=f"EPOCH {epoch} — Test Set")
        
        retain_acc = matrix_val[1, 0]  # Retain, fusion branch
        
        if retain_acc > best_retain_acc:
            best_retain_acc = retain_acc
            best_epoch = epoch
            patience_count = 0

            torch.save(model.state_dict(), save_ckpt_path)

            print(
                f"✓ New best "
                f"(epoch {epoch}, "
                f"retain_acc={retain_acc:.2f}%)"
            )

        else:
            patience_count += 1

            # print(
            #     f"  No improvement. "
            #     f"Patience {patience_count}/{patience}"
            # )
            
            # if patience_count >= patience:
            #     print(f"\n  Early stopping at epoch {epoch}.")
            #     print(f"  Best: epoch {best_epoch}, retain_fusion={best_retain_acc:.2f}%")
            #     break
    
    print("\n" + "="*70)
    print(f"TRAINING COMPLETED!")
    print(f"Best epoch: {best_epoch}  |  Best retain acc: {best_retain_acc:.2f}%")
    print("="*70)
    
    # Load best checkpoint for final evaluation
    print("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(save_ckpt_path, map_location=device))
    
    matrix_after = evaluate_split(model, forget_test_loader, retain_test_loader, device)
    print_accuracy_matrix(matrix_after, title="AFTER UNLEARNING (Test Set)")
    


def parse_args():
    p = argparse.ArgumentParser(description="State of the art Machine Unlearning Methods")

    p.add_argument("--forget-class", type=int, help="head or mid or tail or numeric")
    p.add_argument("--dataset", type=str, help="cifar10 or cifar100 or food101 or places365", default="cifar10")
    p.add_argument("--imb-factor", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()



def main():
    args = parse_args()
    
    forget_class = int(args.forget_class)
    imb_factor = int(args.imb_factor)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = args.dataset
    
    start_time = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    TRAIN_DATA, TEST_DATA, num_classes, num_epochs, TEACHER_PATH, STUDENT_SAVE_DIR, ORACLE_SAVE_DIR, P_AVG_PLOT_DIR, P_K_BAR_PLOT_DIR = process_args(dataset, imb_factor, forget_class, "", "")

    run_neggrad_unlearning(
        TRAIN_DATA=TRAIN_DATA,
        TEST_DATA=TEST_DATA,
        num_classes=num_classes,
        original_ckpt=TEACHER_PATH,
        save_ckpt_path=f"/export/home/achyut/Simarjeet/MUL/Models/Neggrad/{dataset}/neggrad_im{imb_factor}_cls{forget_class}.pth",
        forget_class=forget_class,
        dataset=dataset,
        imb_factor=imb_factor,
        device=device,
        num_workers=0
    )

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Script execution time: {execution_time:.6f} seconds")
    


if __name__ == "__main__":
    main()