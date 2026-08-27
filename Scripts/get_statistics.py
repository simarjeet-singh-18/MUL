import torch
import random
import torchvision
import numpy as  np
from PIL import Image
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset



######################################################################

imb_factor = 1000 # 10 or 100 or 200 or 1000 (for places only)
dataset = "places365" # cifar10 or cifar100 or food101 or places365
test_data_type = "lt" # uniform or lt

######################################################################



if dataset == "cifar10":
    NUM_CLASSES = 10
    if test_data_type == "uniform":
        TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/cifar10"
    elif test_data_type == "lt":
        TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/cifar10/Generated/test_lt_cifar10.pt"
elif dataset == "cifar100":
    NUM_CLASSES = 100
    if test_data_type == "uniform":
        TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/cifar100"
    elif test_data_type == "lt":
        TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/cifar100/Generated/test_lt_cifar100_im200.pt"
elif dataset == "food101":
    NUM_CLASSES = 101
    if test_data_type == "uniform":
        TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/food101/split/val"
    elif test_data_type == "lt":
        TEST_DATA = f"/export/home/achyut/Simarjeet/MUL/Datasets/food101/split/val_im{imb_factor}"
elif dataset == "places365":
    NUM_CLASSES = 365
    if test_data_type == "uniform":
        TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/places365/lt/val"
    elif test_data_type == "lt":
        TEST_DATA = f"/export/home/achyut/Simarjeet/MUL/Datasets/places365/lt/val_im{imb_factor}"

TEACHER_PATH = f"/export/home/achyut/Simarjeet/MUL/Models/Teachers/{dataset}/teacher_im{imb_factor}.pth"
PLOT_DIR = f"/export/home/achyut/Simarjeet/MUL/Plots/{dataset}/statistics/"

learning_rate = 0.001
batch_size = 32
temperature = 4.0
weight_decay = 1e-4



def setup():
    SEED = 18
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    return device, SEED



class SavedDataset(Dataset):

    def __init__(self, file_path, transform=None):
        saved = torch.load(file_path, weights_only=False)
        self.data = saved["data"]
        self.targets = saved["targets"]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label



def data_loaders(dataset):

    if dataset == "cifar10":

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])

    elif dataset == "cifar100":

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761)
            )
        ])
        
    elif dataset == "food101":

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225])
        ])
        
    elif dataset == "places365":

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std  = [0.229, 0.224, 0.225]
            )
        ])
        
    if dataset == "cifar10":
        
        if test_data_type == "uniform":
        
            test_dataset = torchvision.datasets.CIFAR10(
                root=TEST_DATA,
                train=False,
                download=True,
                transform=transform_test
            )
            
        elif test_data_type == "lt":
            
            test_dataset = SavedDataset(
                TEST_DATA,
                transform=transform_test
            )
    
    elif dataset == "cifar100":
        
        if test_data_type == "uniform":

            test_dataset = torchvision.datasets.CIFAR100(
                root=TEST_DATA,
                train=False,
                download=True,
                transform=transform_test
            )
        
        elif test_data_type == "lt":
            
            test_dataset = SavedDataset(
                TEST_DATA,
                transform=transform_test
            )
            
        
    elif dataset == "food101":

        test_dataset = torchvision.datasets.ImageFolder(
            root=TEST_DATA,
            transform=transform_test
        )
        
    elif dataset == "places365":

        test_dataset = torchvision.datasets.ImageFolder(
            root=TEST_DATA,
            transform=transform_test
        )
        
    print("\nClass Distribution (Test):")
    print(Counter(test_dataset.targets))

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return test_loader



def load_models (dataset, device):
    
    if dataset == "cifar10":
    
        teacher = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        
    elif dataset in ["cifar100", "food101", "places365"]:
        
        teacher = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        
    if dataset in ["cifar10", "cifar100"]:
        
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
        NUM_CLASSES
    )

    teacher.load_state_dict(
        torch.load(
            TEACHER_PATH,
            map_location=device
        )
    )
    
    teacher = teacher.to(device)
    teacher.eval()
    
    print("\nTeacher model loaded successfully")
    return teacher



def get_stats(device, teacher):

    class_probs = [[] for _ in range(NUM_CLASSES)]

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            logits = teacher(images)
            probs = torch.softmax(logits, dim=1)

            for i in range(images.size(0)):

                label = labels[i].item()
                class_probs[label].append(
                    probs[i, label].item()
                )
            
    means = np.zeros(NUM_CLASSES)
    vars_ = np.zeros(NUM_CLASSES)
    medians = np.zeros(NUM_CLASSES)

    for c in range(NUM_CLASSES):
        p = np.array(class_probs[c])
        means[c] = p.mean()
        vars_[c] = p.var()
        medians[c] = np.median(p)

    return means, vars_, medians



def get_plots(means, vars_, medians):
    
    plt.figure(figsize=(12,4))
    plt.bar(range(NUM_CLASSES), means)
    plt.xlabel("Class")
    plt.ylabel("Average probability")
    plt.tight_layout()
    plt.savefig(PLOT_DIR + f"class_avg_probability_{test_data_type}.png")
    
    plt.figure(figsize=(12,4))
    plt.bar(range(NUM_CLASSES), vars_)
    plt.xlabel("Class")
    plt.ylabel("Variance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR + f"class_variance_{test_data_type}.png")

    plt.figure(figsize=(12,4))
    plt.bar(range(NUM_CLASSES), medians)
    plt.xlabel("Class")
    plt.ylabel("Median probability")
    plt.tight_layout()
    plt.savefig(PLOT_DIR + f"class_median_{test_data_type}.png")



def mean_cluster(means, SEED):

    X = means.reshape(-1, 1)

    kmeans = KMeans(
        n_clusters=3,
        random_state=SEED,
        n_init=20
    )

    clusters = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.flatten()
    order = np.argsort(centers)

    mapping = {
        order[0]: "Tail",
        order[1]: "Mid",
        order[2]: "Head"
    }

    group_colors = {
        "Head": "green",
        "Mid": "orange",
        "Tail": "red"
    }

    groups = {"Head": [], "Mid": [], "Tail": []}

    plt.figure(figsize=(10,2))

    for i in range(NUM_CLASSES):

        group = mapping[clusters[i]]
        groups[group].append(i)

        plt.scatter(
            means[i],
            0,
            color=group_colors[group],
            s=150
        )

        plt.text(
            means[i],
            0.02,
            str(i),
            fontsize=11,
            ha="center"
        )

    plt.yticks([])
    plt.xlabel("Average Teacher Probability")
    plt.title("KMeans Clustering (Average Probability)")
    plt.grid(True)

    plt.savefig(PLOT_DIR + f"k_mean_mean_{test_data_type}.png", dpi=300)

    print("\n========== Average Probability Clustering ==========")
    for g in ["Head","Mid","Tail"]:
        print(f"{g}: {sorted(groups[g])}")
    


def mvm_cluster(means, vars_, medians, SEED):

    X = np.stack(
        [means, vars_, medians],
        axis=1
    )

    X = StandardScaler().fit_transform(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=SEED,
        n_init=20
    )

    clusters = kmeans.fit_predict(X)

    cluster_mean = []

    for i in range(3):
        cluster_mean.append(means[clusters == i].mean())

    order = np.argsort(cluster_mean)

    mapping = {
        order[0]: "Tail",
        order[1]: "Mid",
        order[2]: "Head"
    }
    
    group_colors = {
        "Head": "green",
        "Mid": "orange",
        "Tail": "red"
    }

    groups = {"Head": [], "Mid": [], "Tail": []}

    plt.figure(figsize=(8,6))

    for i in range(NUM_CLASSES):

        group = mapping[clusters[i]]
        groups[group].append(i)

        plt.scatter(
            means[i],
            vars_[i],
            color=group_colors[group],
            s=140
        )

        plt.text(
            means[i],
            vars_[i],
            str(i),
            fontsize=11
        )

    plt.xlabel("Average Probability")
    plt.ylabel("Variance")
    plt.title("KMeans Clustering (Mean + Variance + Median)")
    plt.grid(True)

    plt.savefig(PLOT_DIR + f"k_mean_3_{test_data_type}.png", dpi=300)

    print("\n========== Mean + Variance + Median Clustering ==========")
    for g in ["Head","Mid","Tail"]:
        print(f"{g}: {sorted(groups[g])}")
        


device, SEED = setup()
test_loader = data_loaders(dataset)
teacher = load_models(dataset, device)

means, vars_, medians = get_stats(device, teacher)
get_plots(means, vars_, medians)
mean_cluster(means, SEED)
mvm_cluster(means, vars_, medians, SEED)