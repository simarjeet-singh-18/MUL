"""
Base Utilities for ADVANCE Unlearning Experiments
==================================================
Common helper functions used across all unlearning methods.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



@torch.no_grad()
def evaluate_split(
    model,
    forget_loader,
    retain_loader,
    device
):

    model.eval()

    def get_acc(loader):

        correct = 0
        total = 0

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return 100 * correct / total

    forget_acc = get_acc(forget_loader)
    retain_acc = get_acc(retain_loader)

    matrix = np.array([
        [forget_acc],
        [retain_acc]
    ])

    return matrix



def print_accuracy_matrix(matrix, title="Accuracy Matrix"):

    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")

    print(f"{'':15} {'Accuracy':>12}")
    print(f"{'-'*50}")

    print(
        f"{'Forget':15} "
        f"{matrix[0,0]:11.2f}%"
    )

    print(
        f"{'Retain':15} "
        f"{matrix[1,0]:11.2f}%"
    )

    print(f"{'='*50}\n")

def save_checkpoint(model, save_path, epoch=0, val_acc=0.0):
    """
    Save model checkpoint.z
    
    Args:
        model: AdvanceMultimodalModel instance
        save_path: Path to save checkpoint
        epoch: Current epoch number
        val_acc: Validation accuracy
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'val_acc': val_acc,
        'model_state_dict': model.state_dict(),
    }, save_path)
    
    print(f"Checkpoint saved -> {save_path}")
    
def load_model(dataset, num_classes):
    
    if dataset == "cifar10":
        model = torchvision.models.resnet18(weights=None)
    elif dataset in ["cifar100", "food101", "places365"]:
        model = torchvision.models.resnet50(weights=None)
        
    if dataset in ["cifar10", "cifar100"]:
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
        
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
    