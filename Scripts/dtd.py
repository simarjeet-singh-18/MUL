# """
# DTD (Descent-to-Delete) Unlearning for ADVANCE
# ===============================================
# Train on retain set with Gaussian noise injection for privacy.

# References:
# - Neel et al. (2021) "Descent-to-Delete"
# """

# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset, Subset
# from PIL import Image
# from torchvision import transforms
# import torchvision
# from tqdm import tqdm

# # from src.dataset import get_forget_splits, get_retain_splits
# # from src.model import AdvanceMultimodalModel
# # from src.labels import NUM_CLASSES

# from soa_utils import evaluate_split, print_accuracy_matrix
# # from mia_v2 import run_mia

# class SavedDataset(Dataset):

#     def __init__(self, file_path, transform=None):
#         saved = torch.load(file_path, weights_only=False)
#         self.data = saved["data"]
#         self.targets = saved["targets"]
#         self.transform = transform

#     def __len__(self):
#         return len(self.targets)

#     def __getitem__(self, idx):
#         image = self.data[idx]
#         label = self.targets[idx]

#         # Convert numpy array to PIL image
#         image = Image.fromarray(image)

#         if self.transform:
#             image = self.transform(image)

#         return image, label
    
# def get_forget_retain_splits(
#     TRAIN_DATA,
#     TEST_DATA,
#     num_classes,
#     forget_class=0,
#     imb_factor=200,
#     val_ratio=0,
#     test_ratio=0.2,
#     seed=42,
#     dataset="cifar100"
# ):
#     """
#     Returns:
#         forget_train, forget_val, forget_test
#         retain_train, retain_val, retain_test
#     """
    
#     # Make transforms for particular datasets
#     if dataset == "cifar10":
        
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 (0.4914, 0.4822, 0.4465),
#                 (0.2023, 0.1994, 0.2010)
#             )
#         ])

#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 (0.4914, 0.4822, 0.4465),
#                 (0.2023, 0.1994, 0.2010)
#             )
#         ])

#     elif dataset == "cifar100":
        
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=(0.5071, 0.4867, 0.4408),
#                 std=(0.2675, 0.2565, 0.2761)
#             )
#         ])

#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=(0.5071, 0.4867, 0.4408),
#                 std=(0.2675, 0.2565, 0.2761)
#             )
#         ])
        
#     elif dataset == "food101":
        
#         transform_train = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(p=0.1),
#             transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
#             transforms.RandomRotation(degrees=15),
#             transforms.RandomGrayscale(p=0.05),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)), 
#         ])

#         transform_test = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225])
#         ])
        
#     elif dataset == "places365":
        
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean = [0.485, 0.456, 0.406],
#                 std  = [0.229, 0.224, 0.225]
#             )
#         ])

#         transform_test = transforms.Compose([
#             transforms.Resize(256),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean = [0.485, 0.456, 0.406],
#                 std  = [0.229, 0.224, 0.225]
#             )
#         ])
    
#     # Load datasets
    
#     if dataset == "cifar10" or dataset == "cifar100":

#         _dataset = SavedDataset(
#             file_path=TRAIN_DATA,
#             transform=transform_train
#         )
        
#     elif dataset == "food101" or dataset == "places365":
        
#         _dataset = torchvision.datasets.ImageFolder(
#             TRAIN_DATA,
#             transform=transform_train
#         )
        
#     if dataset == "cifar10":
    
#         dataset_test = torchvision.datasets.CIFAR10(
#             root=TEST_DATA,
#             train=False,
#             download=True,
#             transform=transform_test
#         )
    
#     elif dataset == "cifar100":

#         dataset_test = torchvision.datasets.CIFAR100(
#             root=TEST_DATA,
#             train=False,
#             download=True,
#             transform=transform_test
#         )
        
#     elif dataset == "food101" or dataset == "places365":
        
#         dataset_test = torchvision.datasets.ImageFolder(
#             root=TEST_DATA,
#             transform=transform_test
#         )

#     targets = np.array(_dataset.targets)
#     targets_test = np.array(dataset_test.targets)

#     # Forget samples
#     forget_idx = np.where(targets == forget_class)[0]
#     forget_idx_test = np.where(targets_test == forget_class)[0]

#     # Retain samples
#     retain_idx = np.where(targets != forget_class)[0]
#     retain_idx_test = np.where(targets_test != forget_class)[0]

#     rng = np.random.default_rng(seed)

#     rng.shuffle(forget_idx)
#     rng.shuffle(retain_idx)
#     rng.shuffle(forget_idx_test)
#     rng.shuffle(retain_idx_test)

#     def split_indices(indices):
#         n = len(indices)

#         n_test = int(n * test_ratio)
#         n_val = int(n * val_ratio)

#         test_idx = indices[:n_test]
#         val_idx = indices[n_test:n_test + n_val]
#         train_idx = indices[n_test + n_val:]

#         return train_idx, val_idx, test_idx

#     forget_train_idx, forget_val_idx, forget_test_idx = split_indices(forget_idx)
#     retain_train_idx, retain_val_idx, retain_test_idx = split_indices(retain_idx)

#     forget_train = Subset(_dataset, forget_train_idx)
#     forget_val = Subset(_dataset, forget_val_idx)
#     forget_test = Subset(dataset_test, forget_idx_test)

#     retain_train = Subset(_dataset, retain_train_idx)
#     retain_val = Subset(_dataset, retain_val_idx)
#     retain_test = Subset(dataset_test, retain_idx_test)

#     return (
#         forget_train,
#         forget_val,
#         forget_test,
#         retain_train,
#         retain_val,
#         retain_test,
#         num_classes
#     )

# def inject_gradient_noise(model, sigma, device):
#     """
#     Inject calibrated Gaussian noise into gradients.
#     Called after backward(), before optimizer.step().
#     """
#     for param in model.parameters():
#         if param.grad is not None:
#             noise = torch.randn_like(param.grad) * sigma
#             param.grad.add_(noise)


# def train_one_epoch_DTD(
#     model,
#     retain_loader,
#     optimizer,
#     device,
#     epoch,
#     sigma,
#     grad_clip=1.0
# ):
#     model.train()

#     criterion = nn.CrossEntropyLoss()

#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0

#     for images, labels in tqdm(
#         retain_loader,
#         desc=f"Epoch {epoch} [DTD]"
#     ):

#         images = images.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(images)

#         loss = criterion(
#             outputs,
#             labels
#         )

#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(
#             model.parameters(),
#             max_norm=grad_clip
#         )

#         inject_gradient_noise(
#             model,
#             sigma,
#             device
#         )

#         optimizer.step()

#         batch_size = labels.size(0)

#         total_loss += loss.item() * batch_size

#         preds = outputs.argmax(dim=1)

#         total_correct += (
#             preds == labels
#         ).sum().item()

#         total_samples += batch_size

#     print(f"\nEpoch {epoch} Summary")
#     print(
#         f"Loss: {total_loss/total_samples:.4f}"
#     )
#     print(
#         f"Acc : {100*total_correct/total_samples:.2f}%"
#     )
#     print(
#         f"Noise σ: {sigma:.8f}"
#     )

# def run_DTD_unlearning(original_ckpt, save_ckpt_path, forget_class,
#                        device, root, lr=1e-4, epochs=15, batch_size=32,
#                        num_workers=4, patience=5, noise_multiplier=0.01,
#                        grad_clip=1.0):
#     """
#     Run DTD unlearning on ADVANCE dataset.
#     """
#     print("\n" + "="*70)
#     print("BUILDING DATASETS")
#     print("="*70)
    
#     # forget_train, forget_val, forget_test = get_forget_splits(
#     #     forget_class=forget_class, val_ratio=0.2, test_ratio=0.1, seed=42
#     # )
#     # retain_train, retain_val, retain_test = get_retain_splits(
#     #     forget_class=forget_class, val_ratio=0.2, test_ratio=0.1, seed=42
#     # )
    
#     forget_train, forget_val, forget_test, retain_train, retain_val, retain_test = get_forget_retain_splits(root=root,forget_class=forget_class)
    
#     print(f"\n  Forget class: {forget_class}")
#     print(f"  Retain train: {len(retain_train)}")
#     print(f"  Retain test : {len(retain_test)}")
    
#     retain_train_loader = DataLoader(retain_train, batch_size=batch_size,
#                                      shuffle=True, num_workers=num_workers)
#     forget_test_loader = DataLoader(forget_test, batch_size=batch_size,
#                                     shuffle=False, num_workers=num_workers)
#     retain_test_loader = DataLoader(retain_test, batch_size=batch_size,
#                                     shuffle=False, num_workers=num_workers)
    
#     print("\n" + "="*70)
#     print("LOADING ORIGINAL MODEL")
#     print("="*70)
    
#     model = torchvision.models.resnet18(
#         weights=None
#     )

#     model.conv1 = nn.Conv2d(
#         3,
#         64,
#         kernel_size=3,
#         stride=1,
#         padding=1,
#         bias=False
#     )

#     model.maxpool = nn.Identity()

#     model.fc = nn.Linear(
#         model.fc.in_features,
#         10
#     )

#     model.load_state_dict(
#         torch.load(
#             original_ckpt,
#             map_location=device
#         )
#     )

#     model = model.to(device)
    
#     print("\nEvaluating BEFORE unlearning...")
#     matrix_before = evaluate_split(model, forget_test_loader, retain_test_loader, device)
#     print_accuracy_matrix(matrix_before, title="BEFORE UNLEARNING")
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
#     print("\n" + "="*70)
#     print("DTD UNLEARNING")
#     print(f"noise_multiplier = {noise_multiplier}")
#     print(f"grad_clip = {grad_clip}")
#     print("="*70)
    
#     best_retain_acc = 0.0
#     best_epoch = 0
#     patience_count = 0
    
#     for epoch in range(1, epochs + 1):
#         print(f"\n{'-'*70}")
#         print(f"Epoch [{epoch}/{epochs}]")
#         print(f"{'-'*70}")
        
#         sigma = noise_multiplier * lr
#         train_one_epoch_DTD(model, retain_train_loader, optimizer, device,
#                            epoch, sigma, grad_clip)
#         scheduler.step()
        
#         matrix_val = evaluate_split(model, forget_test_loader, retain_test_loader, device)
#         print_accuracy_matrix(matrix_val, title=f"EPOCH {epoch}")
        
#         forget_acc = matrix_val[0, 0]
#         retain_acc = matrix_val[1, 0]
        
#         if retain_acc > best_retain_acc:
#             best_retain_acc = retain_acc
#             best_epoch = epoch
#             patience_count = 0
            
#             torch.save(
#                 model.state_dict(),
#                 save_ckpt_path
#             )
            
#             print(f"✓ New best (epoch {epoch}, retain={retain_acc:.2f}%)")
#         else:
#             patience_count += 1
#             # print(f"  No improvement. Patience {patience_count}/{patience}")
#             # if patience_count >= patience:
#             #     print(f"\n  Early stopping at epoch {epoch}.")
#             #     break
    
#     print("\n" + "="*70)
#     print(f"TRAINING COMPLETED! Best: epoch {best_epoch}, acc={best_retain_acc:.2f}%")
#     print("="*70)
    
#     # model = load_model(save_ckpt_path, NUM_CLASSES, device)
    
#     matrix_after = evaluate_split(model, forget_test_loader, retain_test_loader, device)
#     print_accuracy_matrix(matrix_after, title="AFTER UNLEARNING")
    
#     # MIA
#     # print("\n" + "="*70)
#     # print("MIA EVALUATION")
#     # print("="*70)
    
#     # # from src.dataset import RetainDataset, ForgetDataset
#     # full_retain = RetainDataset(forget_class=forget_class)
#     # full_forget = ForgetDataset(forget_class=forget_class)
#     # full_retain_loader = DataLoader(full_retain, batch_size=batch_size,
#     #                                 shuffle=False, num_workers=num_workers)
#     # full_forget_loader = DataLoader(full_forget, batch_size=batch_size,
#     #                                 shuffle=False, num_workers=num_workers)
    
#     # print("\n[Original Model]")
#     # orig_model = load_model(original_ckpt, NUM_CLASSES, device)
#     # run_mia(orig_model, full_retain_loader, retain_test_loader,
#     #         full_forget_loader, device, label="Original")
    
#     # print("\n[Unlearned Model - DTD]")
#     # run_mia(model, full_retain_loader, retain_test_loader,
#     #         full_forget_loader, device, label="Unlearned (DTD)")


# # if __name__ == "__main__":
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     print(f"Device: {device}")
    
# #     ORIGINAL_CKPT = "/home/team2/Unlearning/ADVANCE/models/advance_trained_rerun_01.pth"
# #     SAVE_CKPT_PATH = "/home/team2/Unlearning/ADVANCE/models/unimodal_unlearn/advance_unlearned_dtd.pth"
# #     FORGET_CLASS = "airport"
    
# #     run_DTD_unlearning(
# #         original_ckpt=ORIGINAL_CKPT,
# #         save_ckpt_path=SAVE_CKPT_PATH,
# #         forget_class=FORGET_CLASS,
# #         device=device,
# #         lr=1e-5,
# #         epochs=15,
# #         batch_size=32,
# #         num_workers=4,
# #         patience=3,
# #         noise_multiplier=0.01,
# #         grad_clip=1.0
# #     )

# forget_class = 7
# imb_factor = 200
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data_root = f"/export/home/achyut/Simarjeet/Unlearning/KD-MUL-CIFAR10/Data/Generated/cifar10_lt_im{imb_factor}.pt"


# import time
# import random
# start_time = time.perf_counter()

# SEED = 42

# random.seed(SEED)
# np.random.seed(SEED)

# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

# ORIGINAL_CKPT = f"/export/home/achyut/Simarjeet/Unlearning/KD-MUL-CIFAR10/Models/Teachers/teacher_im{imb_factor}.pth"
# SAVE_CKPT_PATH = f"/export/home/achyut/Simarjeet/Unlearning/KD-MUL-CIFAR10/Models/SOA/DTD/{imb_factor}/dtd_im{imb_factor}_cls{forget_class}.pth"

# run_DTD_unlearning(
#         original_ckpt=ORIGINAL_CKPT,
#         save_ckpt_path=SAVE_CKPT_PATH,
#         forget_class=forget_class,
#         device=device,
#         root=data_root,
#         lr=1e-5,
#         epochs=15,
#         batch_size=32,
#         num_workers=0,
#         patience=10,
#         noise_multiplier=0.01,
#         grad_clip=1.0
#     )

# end_time = time.perf_counter()
# execution_time = end_time - start_time
# print(f"Script execution time: {execution_time:.6f} seconds")

"""
DTD (Descent-to-Delete) Unlearning
==================================
Train on retain set with Gaussian noise injection for privacy.

References:
- Neel et al. (2021) "Descent-to-Delete"
"""

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif dataset == "places365":

        transform_train = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
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



def inject_gradient_noise(model, sigma, device):
    """
    Inject calibrated Gaussian noise into gradients.
    Called after backward(), before optimizer.step().
    """
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * sigma
            param.grad.add_(noise)



def train_one_epoch_DTD(
    model,
    retain_loader,
    optimizer,
    device,
    epoch,
    sigma,
    grad_clip=1.0
):
    model.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(
        retain_loader,
        desc=f"Epoch {epoch} [DTD]"
    ):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip
        )

        inject_gradient_noise(
            model,
            sigma,
            device
        )

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)

        total_correct += (
            preds == labels
        ).sum().item()

        total_samples += batch_size

    print(f"\nEpoch {epoch} Summary")
    print(
        f"Loss: {total_loss/total_samples:.4f}"
    )
    print(
        f"Acc : {100*total_correct/total_samples:.2f}%"
    )
    print(
        f"Noise σ: {sigma:.8f}"
    )



def run_DTD_unlearning(TRAIN_DATA, TEST_DATA, num_classes, original_ckpt, save_ckpt_path, forget_class, imb_factor, dataset,
                       device, lr=1e-5, epochs=15, batch_size=32,
                       num_workers=4, patience=5, noise_multiplier=0.01,
                       grad_clip=1.0):
    """
    Run DTD unlearning.

    Args:
        original_ckpt: Path to original trained model
        save_ckpt_path: Path to save unlearned model
        forget_class: Class to forget
        device: torch device
        lr: Learning rate
        epochs: Max epochs
        batch_size: Batch size
        num_workers: DataLoader workers
        patience: Early stopping patience
        noise_multiplier: Multiplier controlling gradient noise scale
        grad_clip: Gradient clipping max-norm
    """
    print("\n" + "="*70)
    print("BUILDING DATASETS")
    print("="*70)

    # forget_train, forget_val, forget_test = get_forget_splits(
    #     forget_class=forget_class, val_ratio=0.2, test_ratio=0.1, seed=42
    # )
    # retain_train, retain_val, retain_test = get_retain_splits(
    #     forget_class=forget_class, val_ratio=0.2, test_ratio=0.1, seed=42
    # )

    forget_train, forget_val, forget_test, retain_train, retain_val, retain_test, num_classes = get_forget_retain_splits(TRAIN_DATA, TEST_DATA, num_classes, dataset=dataset, imb_factor=imb_factor, forget_class=forget_class)

    print(f"\n  Forget class: {forget_class}")
    print(f"  Retain train: {len(retain_train)}")
    print(f"  Retain test : {len(retain_test)}")

    retain_train_loader = DataLoader(retain_train, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers)
    forget_test_loader = DataLoader(forget_test, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    retain_test_loader = DataLoader(retain_test, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)

    print("\n" + "="*70)
    print("LOADING ORIGINAL MODEL")
    print("="*70)

    # model = torchvision.models.resnet18(weights=None)
    # model.conv1 = nn.Conv2d(
    # 3, 64, kernel_size=3, stride=1, padding=1, bias=False
    # )
    # model.maxpool = nn.Identity()
    # model.fc = nn.Linear(model.fc.in_features, 10)
    model = load_model(dataset, num_classes)
    model.load_state_dict(torch.load(original_ckpt, map_location=device))
    model = model.to(device)

    print("\nEvaluating BEFORE unlearning...")
    matrix_before = evaluate_split(model, forget_test_loader, retain_test_loader, device)
    print_accuracy_matrix(matrix_before, title="BEFORE UNLEARNING")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "="*70)
    print("DTD UNLEARNING")
    print(f"noise_multiplier = {noise_multiplier}")
    print(f"grad_clip = {grad_clip}")
    print("="*70)

    best_retain_acc = 0.0
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, epochs + 1):
        print(f"\n{'-'*70}")
        print(f"Epoch [{epoch}/{epochs}]")
        print(f"{'-'*70}")

        sigma = noise_multiplier * lr
        train_one_epoch_DTD(model, retain_train_loader, optimizer, device,
                            epoch, sigma, grad_clip)
        scheduler.step()

        matrix_val = evaluate_split(model, forget_test_loader, retain_test_loader, device)
        print_accuracy_matrix(matrix_val, title=f"EPOCH {epoch}")

        forget_acc = matrix_val[0, 0]
        retain_acc = matrix_val[1, 0]

        if retain_acc > best_retain_acc:
            best_retain_acc = retain_acc
            best_epoch = epoch
            patience_count = 0

            torch.save(
                model.state_dict(),
                save_ckpt_path
            )

            print(f"✓ New best (epoch {epoch}, retain={retain_acc:.2f}%)")
        else:
            patience_count += 1
            # print(f"  No improvement. Patience {patience_count}/{patience}")
            # if patience_count >= patience:
            #     print(f"\n  Early stopping at epoch {epoch}.")
            #     break

    print("\n" + "="*70)
    print(f"TRAINING COMPLETED! Best: epoch {best_epoch}, acc={best_retain_acc:.2f}%")
    print("="*70)

    # model = load_model(save_ckpt_path, NUM_CLASSES, device)

    matrix_after = evaluate_split(model, forget_test_loader, retain_test_loader, device)
    print_accuracy_matrix(matrix_after, title="AFTER UNLEARNING")



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

    run_DTD_unlearning(
        TRAIN_DATA=TRAIN_DATA,
        TEST_DATA=TEST_DATA,
        num_classes=num_classes,
        original_ckpt=TEACHER_PATH,
        save_ckpt_path=f"/export/home/achyut/Simarjeet/MUL/Models/DTD/{dataset}/dtd_im{imb_factor}_cls{forget_class}.pth",
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