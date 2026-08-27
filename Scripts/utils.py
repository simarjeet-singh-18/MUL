import torch
import torchvision
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset



def process_args(dataset, imb_factor, forget_class, clustering_type, pipeline):
    if dataset == "cifar10":
        TRAIN_DATA = f"/export/home/achyut/Simarjeet/MUL/Datasets/cifar10/Generated/cifar10_lt_im{imb_factor}.pt"
        TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/cifar10"
        num_classes = 10
        num_epochs = 15
    elif dataset == "cifar100":
        TRAIN_DATA = f"/export/home/achyut/Simarjeet/MUL/Datasets/cifar100/Generated/cifar100_lt_im{imb_factor}.pt"
        TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/cifar100"
        num_classes = 100
        num_epochs = 30
    elif dataset == "food101":
        TRAIN_DATA = f"/export/home/achyut/Simarjeet/MUL/Datasets/food101/food-101/split/train_im{imb_factor}"
        TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/food101/food-101/split/val"
        num_classes = 101
        num_epochs = 30
    elif dataset == "places365":
        TRAIN_DATA = f"/export/home/achyut/Simarjeet/MUL/Datasets/places365/lt/train_im{imb_factor}"
        TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/places365/lt/val"
        num_classes = 365
        num_epochs = 30
    
    TEACHER_PATH = f"/export/home/achyut/Simarjeet/MUL/Models/Teachers/{dataset}/teacher_im{imb_factor}.pth"
    STUDENT_SAVE_DIR = f"/export/home/achyut/Simarjeet/MUL/Models/Students/{dataset}/{clustering_type}/{pipeline}_im{imb_factor}_cls{forget_class}.pth"

    P_AVG_PLOT_DIR = f"/export/home/achyut/Simarjeet/MUL/Plots/{dataset}/{clustering_type}/p_avg_im{imb_factor}_cls{forget_class}.png"
    P_K_BAR_PLOT_DIR = f"/export/home/achyut/Simarjeet/MUL/Plots/{dataset}/{clustering_type}/p_k_bar_im{imb_factor}_cls{forget_class}.png"
    
    ORACLE_SAVE_PATH = ""
    
    if pipeline == "orcl":
        ORACLE_SAVE_PATH = f"/export/home/achyut/Simarjeet/MUL/Models/Oracles/{dataset}/{pipeline}_im{imb_factor}_cls{forget_class}.pth"
        

    return TRAIN_DATA, TEST_DATA, num_classes, num_epochs, TEACHER_PATH, STUDENT_SAVE_DIR, ORACLE_SAVE_PATH, P_AVG_PLOT_DIR, P_K_BAR_PLOT_DIR



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
    
    
    
class RemappedDataset(Dataset):
    def __init__(self, subset, forget_class):
        self.subset = subset
        self.forget_class = forget_class

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]

        
        if (label > self.forget_class):
            label = label - 1

        return image, label



def data_loaders(dataset, TRAIN_DATA, TEST_DATA, batch_size, pipeline, forget_class):

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
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
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
        
    if dataset == "cifar10":
        
        train_dataset = SavedDataset(
            TRAIN_DATA,
            transform=transform_train
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=TEST_DATA,
            train=False,
            download=True,
            transform=transform_test
        )
    
    elif dataset == "cifar100":
        
        train_dataset = SavedDataset(
            TRAIN_DATA,
            transform=transform_train
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root=TEST_DATA,
            train=False,
            download=True,
            transform=transform_test
        )
        
    elif dataset == "food101":
        
        train_dataset = torchvision.datasets.ImageFolder(
            TRAIN_DATA,
            transform=transform_train
        )

        test_dataset = torchvision.datasets.ImageFolder(
            root=TEST_DATA,
            transform=transform_test
        )
        
    elif dataset == "places365":
        
        train_dataset = torchvision.datasets.ImageFolder(
            TRAIN_DATA,
            transform=transform_train
        )

        test_dataset = torchvision.datasets.ImageFolder(
            root=TEST_DATA,
            transform=transform_test
        )
        
    print("\nClass Distribution:")
    print(Counter(train_dataset.targets))
    
    if pipeline == "orcl":
        train_indices = [i for i, label in enumerate(train_dataset.targets) if label != forget_class]
        test_indices = [i for i, label in enumerate(test_dataset.targets) if label != forget_class]

        train_dataset_filtered = Subset(train_dataset, train_indices)
        test_dataset_filtered = Subset(test_dataset, test_indices)
        
        train_dataset_filtered = RemappedDataset(train_dataset_filtered, forget_class)
        test_dataset_filtered = RemappedDataset(test_dataset_filtered, forget_class)
        
        train_dataset = train_dataset_filtered
        test_dataset = test_dataset_filtered
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader
