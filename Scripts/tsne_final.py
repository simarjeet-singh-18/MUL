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
import numpy as  np
import random

SEED = 12
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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


batch_size = 128
learning_rate = 0.0001
num_epochs = 30
temperature = 4.0
imb_factor = 200
forget_class = 0
weight_decay = 1e-4


head_group = [index for index in range(3)]
mid_group = [index for index in range(3, 7)]
tail_group = [index for index in range(7, 10)]


if forget_class in head_group:
    forget_group = "Head"
elif forget_class in mid_group:
    forget_group = "Mid"
else:
    forget_group = "Tail"


test_dataset = torchvision.datasets.CIFAR10(
    root="MUL/Datasets/cifar10",
    train=False,
    download=True,
    transform=transform_test
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# Load teacher

teacher = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

teacher.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

teacher.maxpool = nn.Identity()

teacher.fc = nn.Linear(
    teacher.fc.in_features,
    10
)

teacher.load_state_dict(
    torch.load(
        f"MUL/Models/Teachers/cifar10/teacher_im{imb_factor}.pth",
        map_location=device
    )
)

teacher = teacher.to(device)
teacher.eval()

for param in teacher.parameters():
    param.requires_grad = False

print("\nTeacher model loaded successfully")

# Load student normal

student = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

student.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

student.maxpool = nn.Identity()

student.fc = nn.Linear(
    student.fc.in_features,
    10
)

student = student.to(device)
student.load_state_dict(
    torch.load(
        f"MUL/Models/Students/cifar10/manual/nor_im{imb_factor}_clshead.pth",
        map_location=device
    )
)

print("\nStudent normal model created")


# Load student imp

student_imp = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

student_imp.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

student_imp.maxpool = nn.Identity()

student_imp.fc = nn.Linear(
    student_imp.fc.in_features,
    10
)

student_imp = student_imp.to(device)
student_imp.load_state_dict(
    torch.load(
        f"MUL/Models/Students/cifar10/manual/imp_im{imb_factor}_clshead.pth",
        map_location=device
    )
)


print("\nStudent imp model created")

# Load oracle

oracle = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

oracle.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

oracle.maxpool = nn.Identity()

oracle.fc = nn.Linear(
    oracle.fc.in_features,
    9
)

oracle = oracle.to(device)
oracle.load_state_dict(
    torch.load(
        f"MUL/Models/Oracles/cifar10/orcl_im{imb_factor}_clshead.pth",
        map_location=device
    )
)

# Neggrad

neggrad = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

neggrad.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

neggrad.maxpool = nn.Identity()

neggrad.fc = nn.Linear(
    neggrad.fc.in_features,
    10
)

neggrad = neggrad.to(device)
neggrad.load_state_dict(
    torch.load(
        f"MUL/Models/Neggrad/cifar10/neggrad_im{imb_factor}_clshead.pth",
        map_location=device
    )
)

# DTD

dtd = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

dtd.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

dtd.maxpool = nn.Identity()

dtd.fc = nn.Linear(
    dtd.fc.in_features,
    10
)

dtd = dtd.to(device)
dtd.load_state_dict(
    torch.load(
        f"MUL/Models/DTD/cifar10/dtd_im{imb_factor}_clshead.pth",
        map_location=device
    )
)

# lcodec

lcodec = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

lcodec.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

lcodec.maxpool = nn.Identity()

lcodec.fc = nn.Linear(
    lcodec.fc.in_features,
    10
)

lcodec = lcodec.to(device)
lcodec.load_state_dict(
    torch.load(
        f"MUL/Models/Students/cifar10/manual/lcodec_im{imb_factor}_clshead.pth",
        map_location=device
    )
)

# ul

ul = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

ul.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

ul.maxpool = nn.Identity()

ul.fc = nn.Linear(
    ul.fc.in_features,
    10
)

ul = ul.to(device)
ul.load_state_dict(
    torch.load(
        f"MUL/Models/Students/cifar10/manual/ul_im{imb_factor}_clshead.pth",
        map_location=device
    )
)

# scrub

scrub = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

scrub.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

scrub.maxpool = nn.Identity()

scrub.fc = nn.Linear(
    scrub.fc.in_features,
    10
)

scrub = scrub.to(device)
scrub.load_state_dict(
    torch.load(
        f"MUL/Models/Students/cifar10/manual/scrub_im{imb_factor}_clshead.pth",
        map_location=device
    )
)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


num_samples = 2000

classes = [
    "airplane (forget)",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

class FeatureExtractor(torch.nn.Module):

    def __init__(self, model):

        super().__init__()

        self.features = torch.nn.Sequential(
            *list(model.children())[:-1]
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        return x

teacher_feature_model = FeatureExtractor(teacher).to(device)
student_feature_model = FeatureExtractor(student).to(device)
student_imp_feature_model = FeatureExtractor(student_imp).to(device)
oracle_feature_model = FeatureExtractor(oracle).to(device)
neggrad_feature_model = FeatureExtractor(neggrad).to(device)
dtd_feature_model = FeatureExtractor(dtd).to(device)
lcodec_feature_model = FeatureExtractor(lcodec).to(device)
ul_feature_model = FeatureExtractor(ul).to(device)
scrub_feature_model = FeatureExtractor(scrub).to(device)


teacher_feature_model.eval()
student_feature_model.eval()
student_imp_feature_model.eval()
oracle_feature_model.eval()
neggrad_feature_model.eval()
dtd_feature_model.eval()
lcodec_feature_model.eval()
ul_feature_model.eval()
scrub_feature_model.eval()

teacher_features = []
student_features = []
student_imp_features = []
oracle_features = []
neggrad_features = []
dtd_features = []
lcodec_features = []
ul_features = []
scrub_features = []
all_labels = []

count = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        teacher_feat = teacher_feature_model(images)
        student_feat = student_feature_model(images)
        student_imp_feat = student_imp_feature_model(images)
        oracle_feat = oracle_feature_model(images)
        neggrad_feat = neggrad_feature_model(images)
        dtd_feat = dtd_feature_model(images)
        lcodec_feat = lcodec_feature_model(images)
        ul_feat = ul_feature_model(images)
        scrub_feat = scrub_feature_model(images)

        teacher_features.append(
            teacher_feat.cpu()
        )

        student_features.append(
            student_feat.cpu()
        )

        student_imp_features.append(
            student_imp_feat.cpu()
        )
        
        oracle_features.append(
            oracle_feat.cpu()
        )

        neggrad_features.append(
            neggrad_feat.cpu()
        )

        dtd_features.append(
            dtd_feat.cpu()
        )
        
        lcodec_features.append(
            lcodec_feat.cpu()
        )
        
        ul_features.append(
            ul_feat.cpu()
        )
        
        scrub_features.append(
            scrub_feat.cpu()
        )

        all_labels.append(labels)

        count += labels.size(0)

        if count >= num_samples:
            break

teacher_features = torch.cat(
    teacher_features,
    dim=0
)[:num_samples]

student_features = torch.cat(
    student_features,
    dim=0
)[:num_samples]

student_imp_features = torch.cat(
    student_imp_features,
    dim=0
)[:num_samples]

oracle_features = torch.cat(
    oracle_features,
    dim=0
)[:num_samples]

neggrad_features = torch.cat(
    neggrad_features,
    dim=0
)[:num_samples]

dtd_features = torch.cat(
    dtd_features,
    dim=0
)[:num_samples]

lcodec_features = torch.cat(
    lcodec_features,
    dim=0
)[:num_samples]

ul_features = torch.cat(
    ul_features,
    dim=0
)[:num_samples]

scrub_features = torch.cat(
    scrub_features,
    dim=0
)[:num_samples]

all_labels = torch.cat(
    all_labels,
    dim=0
)[:num_samples]

teacher_features = teacher_features.numpy()
student_features = student_features.numpy()
student_imp_features = student_imp_features.numpy()
oracle_features = oracle_features.numpy()
neggrad_features = neggrad_features.numpy()
dtd_features = dtd_features.numpy()
lcodec_features = lcodec_features.numpy()
ul_features = ul_features.numpy()
scrub_features = scrub_features.numpy()

all_labels = all_labels.numpy()

print("Feature extraction complete")


print("Running Teacher t-SNE...")

teacher_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
).fit_transform(teacher_features)

print("Running Student Nor t-SNE...")

student_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
).fit_transform(student_features)

print("Running Student Imp t-SNE...")

student_imp_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
).fit_transform(student_imp_features)

print("Oracle t-SNE...")

oracle_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
).fit_transform(oracle_features)

print("Neggrad t-SNE...")

neggrad_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
).fit_transform(neggrad_features)

print("DTD t-SNE...")

dtd_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
).fit_transform(dtd_features)

print("LCodec t-SNE...")

lcodec_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
).fit_transform(lcodec_features)

print("UL t-SNE...")

ul_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
).fit_transform(ul_features)

print("Scrub t-SNE...")

scrub_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
).fit_transform(scrub_features)

print("t-SNE complete")

fig = plt.figure(figsize=(40, 16))

gs = GridSpec(2, 10, figure=fig)

panel_specs = [
    (teacher_tsne,     "Target",   0, 0),
    (student_tsne,     "KD-Based", 0, 2),
    (student_imp_tsne, "Proposed", 0, 4),
    (oracle_tsne,      "Oracle",   0, 6),
    (neggrad_tsne,     "Neggrad",  0, 8),
    (dtd_tsne,         "DTD",      1, 1),
    (lcodec_tsne,      "LCodec",   1, 3),
    (ul_tsne,          "UL",       1, 5),
    (scrub_tsne,       "Scrub",    1, 7),
]

first_ax = None

for data, title, row, cstart in panel_specs:

    ax = fig.add_subplot(gs[row, cstart:cstart + 2])

    if first_ax is None:
        first_ax = ax

    for class_idx in range(10):

        indices = all_labels == class_idx

        ax.scatter(
            data[indices, 0],
            data[indices, 1],
            s=10,
            alpha=0.7,
            label=classes[class_idx]
        )

    ax.set_title(
        title,
        fontsize=25
    )

    ax.tick_params(
        axis='both',
        labelsize=17
    )


handles, labels_legend = first_ax.get_legend_handles_labels()

class_legend = fig.legend(
    handles,
    labels_legend,
    loc="upper center",
    bbox_to_anchor=(0.41, 0.97),
    ncol=5,
    fontsize=20,
    markerscale=4
)

fig.add_artist(class_legend) 

imb_handle = [
    plt.Line2D([0], [0], linestyle='none')  
]

imb_legend = fig.legend(
    handles=imb_handle,
    labels=[f"Imbalance Factor: {imb_factor}"],
    loc="upper right",
    bbox_to_anchor=(0.85, 0.965),
    fontsize=22,
    frameon=True,
    handlelength=0,
    handletextpad=0
)

plt.subplots_adjust(
    top=0.85,
    wspace=0.25,
    hspace=0.30
)

plt.savefig(
    f"tsne_im{imb_factor}_{forget_group.lower()}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()