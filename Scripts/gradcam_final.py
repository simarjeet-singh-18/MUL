from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random

import cv2
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_classes  = 101
forget_class = 2         
counter = 0
wait = 2

target_path          = "MUL/Models/Teachers/food101/teacher_im200.pth"                
normal_method_path   = "MUL/Models/Students/food101/manual/nor_im200_clshead.pth"    
proposed_method_path = "MUL/Models/Students/food101/manual/imp_im200_clshead.pth"    
oracle_path          = "MUL/Models/Oracles/food101/orcl_im200_clshead.pth"           
neggrad_path         = "MUL/Models/Neggrad/food101/neggrad_im200_clshead.pth"        
dtd_path             = "MUL/Models/DTD/food101/dtd_im200_clshead.pth"                
lcodec_path          = "MUL/Models/Students/food101/manual/lcodec_im200_clshead.pth" 
ul_path              = "MUL/Models/Students/food101/manual/ul_im200_clshead.pth"     
scrub_path           = "MUL/Models/Students/food101/manual/scrub_im200_clshead.pth"  

model_specs = [
    ("Target",   target_path),
    ("Oracle",   oracle_path),
    ("KD-Based", normal_method_path),
    ("Proposed", proposed_method_path),
    ("Neggrad",  neggrad_path),
    ("DTD",      dtd_path),
    ("LCodec",   lcodec_path),
    ("UL",       ul_path),
    ("Scrub",    scrub_path),        
]

n_rows, n_cols = 1, 9

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = torchvision.datasets.ImageFolder(
    root="MUL/Datasets/food101/food-101/split/val",
    transform=transform_test
)

classes = test_dataset.classes

print(f"Forget class: {classes[forget_class]}")

def load_resnet50(ckpt_path, name):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if name == "Oracle":
        model.fc = nn.Linear(model.fc.in_features, num_classes - 1)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(
            self.save_activation
        )

        self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class, oracle=False):

        with torch.enable_grad():
            output = self.model(input_tensor)

        self.model.zero_grad()

        tgt = output.argmax(1).item() if oracle else target_class
        target = output[0, tgt]

        target.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        weights = torch.mean(
            gradients,
            dim=(1, 2)
        )

        cam = torch.zeros(
            activations.shape[1:],
            device=device
        )

        for i, w in enumerate(weights):

            cam += w * activations[i]

        cam = torch.relu(cam)

        cam -= cam.min()

        cam /= (cam.max() + 1e-8) 

        cam = cam.detach().cpu().numpy()

        cam = np.nan_to_num(cam, nan=0.0) 

        return cam


print("\nLoading models...")
models_list = []
for name, path in model_specs:
    if path:
        print(f"Loading: {name}")
        m = load_resnet50(path, name)
        gc = GradCAM(m, m.layer4[-1])
        models_list.append((name, m, gc))
    else:
        print(f"Skipping (no path): {name}")
        models_list.append((name, None, None))
print("Model loading done")

# ============================================================
# FIND AN IMAGE OF THE FORGET CLASS
# ============================================================

target_image = None
target_label = None

for image, label in test_dataset:

    if label == forget_class:

        if counter <= wait:
            counter += 1
            continue

        target_image = image.unsqueeze(0).to(device)
        target_label = label
        break

print("Images found")

# ============================================================
# ORIGINAL IMAGE
# ============================================================

image_np = target_image[0].cpu().numpy()

# CHW -> HWC
image_np = np.transpose(
    image_np,
    (1, 2, 0)
)

# Unnormalize  (must match the ImageNet stats used in transform_test)
mean = np.array([0.485, 0.456, 0.406])

std = np.array([0.229, 0.224, 0.225])

image_np = std * image_np + mean

image_np = np.clip(image_np, 0, 1)

# ============================================================
# HELPER: CAM -> OVERLAY
# ============================================================

def make_overlay(cam, image_np):

    if torch.is_tensor(cam):
        cam = cam.detach().cpu().numpy()

    # cv2 doesn't like float64 or non-contiguous arrays
    cam = np.ascontiguousarray(cam.astype(np.float32))

    cam = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    heatmap = heatmap / 255.0

    overlay = 0.6 * image_np + 0.4 * heatmap

    return np.clip(overlay, 0, 1)

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(n_rows, n_cols, figsize=(45, 6))
axes = axes.ravel()

for ax, (name, model, gradcam) in zip(axes, models_list):

    if model is None:
        # No path provided -> leave this panel blank
        ax.axis("off")
        continue

    cam = gradcam.generate(target_image, forget_class, oracle=(name == "Oracle"))

    overlay = make_overlay(cam, image_np)

    ax.imshow(overlay)

    ax.set_title(name, fontsize=30)

    ax.axis("off")

plt.tight_layout()

# save BEFORE show (show clears the figure, so savefig-after gives a blank file)
plt.savefig("gradcam.png")

plt.show()