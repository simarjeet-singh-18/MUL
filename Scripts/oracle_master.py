import time
import torch
import random
import argparse
import numpy as  np
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import data_loaders, process_args


def setup(dataset, forget_class, clustering_type, pipeline, SEED):
    
    SEED = SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    clusters = {

        "cifar10" : {

            "manual" : {
                "head" : [index for index in range(3)],
                "mid" : [index for index in range(3, 7)],
                "tail" : [index for index in range(7, 10)]
            },
            
            "mean" : {
                "head" : [0, 1],
                "mid" : [2, 3, 4, 7],
                "tail" : [5, 6, 8, 9]
            },
            
            "mvm" : {
                "head" : [0, 1],
                "mid" : [2, 3, 4, 7],
                "tail" : [5, 6, 8, 9]
            }
        },
        
        "cifar100" : {
            
            "manual" : {
                "head" : [index for index in range(0, 21)],
                "mid" : [index for index in range(21, 61)],
                "tail" : [index for index in range(61, 100)]
            },
            
            "mean" : {
                "head" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 28, 30, 33, 36, 39, 41, 48, 53],
                "mid" : [11, 18, 25, 26, 27, 29, 31, 32, 34, 35, 37, 38, 40, 42, 43, 47, 49, 51, 52, 54, 56, 57, 58, 60, 61, 62, 69, 76, 82],
                "tail" : [44, 45, 46, 50, 55, 59, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
            },
            
            "mvm" : {
                "head" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 29, 30, 33, 36, 39, 41, 48, 53, 56, 60, 82],
                "mid" : [11, 25, 27, 31, 32, 34, 35, 37, 38, 40, 42, 43, 44, 47, 49, 51, 52, 54, 57, 58, 61, 62, 63, 67, 69, 74, 76, 94],
                "tail" : [45, 46, 50, 55, 59, 64, 65, 66, 68, 70, 71, 72, 73, 75, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99]  
            }

        },
        
        "food101" : {
            
            "manual" : {
                "head" : [index for index in range(0, 25)],
                "mid" : [index for index in range(25, 50)],
                "tail" : [index for index in range(50, 101)]
            },
            
            "mean" : {
                "head" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 38, 40, 44, 45, 54, 69],
                "mid" : [22, 26, 31, 36, 37, 39, 41, 42, 43, 46, 47, 48, 49, 51, 52, 53, 59, 60, 61, 63, 64, 65, 68, 70, 75, 81],
                "tail" : [50, 55, 56, 57, 58, 62, 66, 67, 71, 72, 73, 74, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
            },
            
            "mvm" : {
                "head" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 40, 41, 44, 45, 54, 63, 64, 69],
                "mid" : [26, 31, 37, 39, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 55, 59, 60, 61, 62, 65, 68, 70, 71, 75, 76, 78, 79, 81, 88, 91],
                "tail" : [56, 57, 58, 66, 67, 72, 73, 74, 77, 80, 82, 83, 84, 85, 86, 87, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100]
            }
            
        },
        
        "places365" : {
            
            "manual" : {
                "head" : [index for index in range(0, 75)],
                "mid" : [index for index in range(75, 250)],
                "tail" : [index for index in range(250, 365)]
            },
            
            "mean" : {
                "head" : [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 24, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 40, 41, 42, 44, 45, 47, 48, 49, 55, 58, 59, 60, 61, 64, 65, 68, 70, 71, 72, 73, 74, 76, 82, 83, 86, 89, 95, 98, 105, 111, 112, 118, 135, 145, 152, 157],
                "mid" : [8, 12, 19, 20, 22, 23, 25, 28, 32, 38, 39, 43, 46, 50, 51, 52, 53, 54, 56, 57, 62, 63, 66, 69, 77, 78, 79, 81, 84, 85, 87, 88, 90, 92, 93, 94, 96, 100, 102, 104, 106, 107, 108, 110, 113, 116, 117, 119, 121, 122, 123, 124, 125, 126, 129, 131, 133, 136, 137, 140, 141, 142, 143, 144, 147, 149, 150, 151, 155, 156, 158, 159, 163, 164, 165, 166, 167, 168, 169, 170, 173, 175, 176, 177, 179, 180, 184, 189, 190, 191, 195, 199, 201, 207, 208, 225, 247, 251, 263, 276, 277],
                "tail" : [21, 67, 75, 80, 91, 97, 99, 101, 103, 109, 114, 115, 120, 127, 128, 130, 132, 134, 138, 139, 146, 148, 153, 154, 160, 161, 162, 171, 172, 174, 178, 181, 182, 183, 185, 186, 187, 188, 192, 193, 194, 196, 197, 198, 200, 202, 203, 204, 205, 206, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364]
            },
            
            "mvm" : {
                "head" : [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 24, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 47, 48, 49, 51, 55, 57, 58, 59, 60, 61, 63, 64, 65, 68, 70, 71, 72, 73, 74, 76, 79, 82, 83, 86, 89, 95, 98, 100, 104, 105, 111, 112, 113, 118, 135, 145, 147, 152, 155, 157],
                "mid" : [8, 12, 19, 20, 22, 23, 25, 28, 32, 39, 43, 46, 50, 52, 53, 54, 56, 62, 66, 69, 77, 78, 81, 84, 85, 87, 88, 90, 92, 93, 94, 96, 102, 103, 106, 107, 108, 110, 116, 117, 119, 121, 122, 123, 124, 125, 126, 129, 131, 133, 136, 137, 140, 141, 142, 143, 144, 149, 150, 151, 154, 156, 158, 159, 163, 164, 165, 166, 167, 168, 169, 170, 173, 175, 176, 177, 179, 180, 184, 188, 189, 190, 191, 195, 199, 201, 207, 208, 211, 225, 247, 251, 255, 263, 276, 277, 294],
                "tail" : [21, 67, 75, 80, 91, 97, 99, 101, 109, 114, 115, 120, 127, 128, 130, 132, 134, 138, 139, 146, 148, 153, 160, 161, 162, 171, 172, 174, 178, 181, 182, 183, 185, 186, 187, 192, 193, 194, 196, 197, 198, 200, 202, 203, 204, 205, 206, 209, 210, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 252, 253, 254, 256, 257, 258, 259, 260, 261, 262, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364]
            }
            
        }
        
    }
    
    def first_common(listA, listB, listC):
        setA = set(listB)
        setB = set(listC)
        
        first_common = next((item for item in listA if item in setA and item in setB), None)
        
        if first_common == None:
            raise ValueError("No common element found in clusters")
        
        return first_common
    
    
    head_group = clusters[dataset][clustering_type]["head"]
    mid_group = clusters[dataset][clustering_type]["mid"]
    tail_group = clusters[dataset][clustering_type]["tail"]
            
    if isinstance(forget_class, str):
        if forget_class == "head":
            forget_class = first_common(clusters[dataset]["manual"]["head"], clusters[dataset]["mean"]["head"], clusters[dataset]["mvm"]["head"])
        elif forget_class == "mid":
            forget_class = first_common(clusters[dataset]["manual"]["mid"], clusters[dataset]["mean"]["mid"], clusters[dataset]["mvm"]["mid"])
        elif forget_class == "tail":
            forget_class = first_common(clusters[dataset]["manual"]["tail"], clusters[dataset]["mean"]["tail"], clusters[dataset]["mvm"]["tail"])
        else:
            # Numeric string such as "5"
            forget_class = int(forget_class)
    
    forget_class = int(forget_class)


        
    print("Dataset ", dataset)
    print("Clustering ", clustering_type)
    print("Forget class ", forget_class)
    print("Pipeline ", pipeline)
    print("\nHead group ", head_group)
    print("Mid group ", mid_group)
    print("Tail group ", tail_group)
        
    return device, head_group, mid_group, tail_group, forget_class



def load_models (dataset, device, num_classes, TEACHER_PATH):
    
    if dataset == "cifar10":
    
        teacher = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        
        student = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        
    elif dataset in ["cifar100", "food101", "places365"]:
        
        teacher = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        
        student = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
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
        
        student.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        teacher.maxpool = nn.Identity()
        student.maxpool = nn.Identity()
    
        

    teacher.fc = nn.Linear(
        teacher.fc.in_features,
        num_classes
    )
    
    student.fc = nn.Linear(
        student.fc.in_features,
        num_classes
    )

    teacher.load_state_dict(
        torch.load(
            TEACHER_PATH,
            map_location=device
        )
    )

    student.load_state_dict(teacher.state_dict())

    teacher = teacher.to(device)
    student = student.to(device)
    
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    print("\nTeacher model loaded successfully")
    print("\nStudent model created")
    return teacher, student
   


def plot_p_avg(p_avg, num_epochs, P_AVG_PLOT_DIR):
    epochs = np.arange(1, num_epochs + 1)
    labels = ['Head', 'Middle', 'Tail']
    markers = ['o', 's', '^']
    
    fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    fig.set_figwidth(6.5)

    for i in range(3):
        axes.plot(
            epochs,
            p_avg[:, i].cpu().numpy(),
            marker=markers[i],
            linewidth=1.8,
            markersize=4,
            label=labels[i]
        )

    axes.set_xlabel('Epochs', fontsize=17)
    axes.set_ylabel('Group view probabilities', fontsize=17)
    axes.set_title(r'Figure 1: $P_{avg}$', y=-0.3, fontsize=20)
    axes.legend(fontsize=15)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(P_AVG_PLOT_DIR, dpi=300, bbox_inches="tight")
    


def plot_p_k_bar(p_k_bar, num_epochs, P_K_BAR_PLOT_DIR):
    epochs = np.arange(1, num_epochs + 1)
    labels = ['Head', 'Middle', 'Tail']
    markers = ['o', 's', '^']

    fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    fig.set_figwidth(6.5)

    for i in range(3):
        axes.plot(
            epochs,
            p_k_bar[:, i].cpu().numpy(),
            marker=markers[i],
            linewidth=1.8,
            markersize=4,
            label=labels[i]
        )

    axes.set_xlabel('Epochs', fontsize=17)
    axes.set_ylabel('Group view probabilities', fontsize=17)
    axes.set_title(r'Figure 2: $\overline{P_k}$', y=-0.3, fontsize=20)
    axes.legend(fontsize=15)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(P_K_BAR_PLOT_DIR, dpi=300, bbox_inches="tight")
    
    
    
def train(device, learning_rate, batch_size, temperature, weight_decay, head_group, mid_group, tail_group, train_loader, test_loader, teacher, student, forget_class, num_epochs, num_classes, pipeline, STUDENT_SAVE_DIR, P_AVG_PLOT_DIR, P_K_BAR_PLOT_DIR):
    
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    optimizer = optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    p_avg_plot = torch.empty(0, 3, device=device)
    p_k_bar_plot = torch.empty(0, 3, device=device)

    start_time = time.perf_counter()

    for epoch in range(num_epochs):

        student.train()
        p_avg_log = torch.empty(0, 3, device=device)
        p_k_bar_log = torch.empty(0, num_classes, device=device)

        for batch_id, (images, labels) in enumerate(train_loader):
        
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
        
            # Make predictions
            prediction_teacher = teacher(images)
            prediction_student = student(images)
        
            # Calculate probabilities
            prob_student = torch.log_softmax(prediction_student/temperature, dim=1)
            prob_teacher = torch.softmax(prediction_teacher/temperature, dim=1)
    
            # Improvised pipeline
        
            # Get group wise probabilities
            if pipeline == "imp":
                
                # Get groups
                prob_head = prob_teacher[:, head_group]
                prob_mid  = prob_teacher[:, mid_group]
                prob_tail = prob_teacher[:, tail_group]
    
                # Average them
                prob_head_avg = prob_head.sum() / batch_size
                prob_mid_avg = prob_mid.sum() / batch_size
                prob_tail_avg = prob_tail.sum() / batch_size
        
                # Make tensors
                prob_avg = torch.tensor([prob_head_avg, prob_mid_avg, prob_tail_avg], device=device)
                prob_avg_ex = torch.tensor([1/3, 1/3, 1/3], device=device)
        
                # Calculate scalaing factors
                s_f = prob_avg_ex / prob_avg
        
                # Scale group probabilities
                scaled_head = prob_head * s_f[0]
                scaled_mid = prob_mid * s_f[1]
                scaled_tail = prob_tail * s_f[2]
        
                # Normalization
                prob_k = torch.zeros_like(prob_teacher)

                prob_k[:, head_group] = scaled_head
                prob_k[:, mid_group]  = scaled_mid
                prob_k[:, tail_group] = scaled_tail
                prob_k_bar = prob_k / prob_k.sum(dim=1, keepdim=True)
        
                # Masking
                mask = torch.ones_like(prob_k_bar)
                mask[:, forget_class] = -torch.inf
                masked_probs = prob_k_bar * mask
                prob_u = torch.softmax(masked_probs, dim=1)
                
                p_avg_log = torch.cat([p_avg_log, prob_avg.unsqueeze(0)])
                p_k_bar_log = torch.cat([p_k_bar_log, prob_k_bar])
                
            elif pipeline == "nor":
                
                mask = torch.ones_like(prob_teacher)
                mask[:, forget_class] = -torch.inf
                masked_probs = prob_teacher * mask
                prob_u = torch.softmax(masked_probs, dim=1)
        
            loss = kl_loss(prob_student, prob_u)
            loss.backward()
            optimizer.step()

        length = p_avg_log.shape[0]
        p_avg_log = p_avg_log.sum(dim=0) / length
    
        length = p_k_bar_log.shape[0]
        p_k_bar_log = p_k_bar_log.sum(dim=0) / length
    
        p_k_bar_head = p_k_bar_log[head_group]
        p_k_bar_mid = p_k_bar_log[mid_group]
        p_k_bar_tail = p_k_bar_log[tail_group]
    
        p_k_bar_head = p_k_bar_head.sum() / len(p_k_bar_head)
        p_k_bar_mid = p_k_bar_mid.sum() / len(p_k_bar_mid)
        p_k_bar_tail = p_k_bar_tail.sum() / len(p_k_bar_tail)
    
        p_k_bar_log = torch.tensor([p_k_bar_head.item() * len(head_group), p_k_bar_mid.item() * len(mid_group), p_k_bar_tail.item() * len(tail_group)])
    
        p_k_bar_log = p_k_bar_log.to(device=device)
        p_avg_plot = torch.cat([p_avg_plot, p_avg_log.unsqueeze(0)])
        p_k_bar_plot = torch.cat([p_k_bar_plot, p_k_bar_log.unsqueeze(0)])

    
        student.eval()
    
        correct = 0
        total = 0
    
        class_correct = [0 for _ in range(num_classes)]
        class_total = [0 for _ in range(num_classes)]

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = student(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
            
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1
            
                correct += (predicted == labels).sum().item()
    
        accuracy = 0
    
        for i in range(num_classes):
            acc = 100 * class_correct[i] / class_total[i]
            print(f"{i}: {acc:.2f}%")

            if (i != forget_class): accuracy += acc
        
        accuracy /= num_classes - 1
    
        print(
            f"\nEpoch [{epoch+1}/{num_epochs}] "
            f"Retain Accuracy: {accuracy:.2f}%"
        )


    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Script execution time: {execution_time:.6f} seconds")
    print("\nTraining Finished")
    
    
    torch.save(
        student.state_dict(),
        STUDENT_SAVE_DIR
    )
    
    if pipeline == "imp":
        plot_p_avg (p_avg_plot, num_epochs, P_AVG_PLOT_DIR)
        plot_p_k_bar(p_k_bar_plot, num_epochs, P_K_BAR_PLOT_DIR)
        
        

def parse_args():
    p = argparse.ArgumentParser(description="Machine Unlearning")

    p.add_argument("--imb-factor", default="200")
    p.add_argument("--dataset", default="cifar10", help="cifar10 or cifar100 or food101 or places365")
    p.add_argument("--forget-class", default="head", help="head or mid or tail or numeric")
    # p.add_argument("--clustering-type", default="manual", help="manual or mean or mvm")
    # p.add_argument("--pipeline", default="imp", help="imp or nor")
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()



def main():
    args = parse_args()
    TRAIN_DATA, TEST_DATA, num_classes, num_epochs, TEACHER_PATH, STUDENT_SAVE_DIR, P_AVG_PLOT_DIR, P_K_BAR_PLOT_DIR = process_args(args.dataset, args.imb_factor, args.forget_class, args.clustering_type, args.pipeline)
    
    learning_rate = 0.001
    batch_size = 32
    temperature = 4.0
    weight_decay = 1e-4

    
    device, head_group, mid_group, tail_group, forget_class = setup(args.dataset, args.forget_class, args.clustering_type, args.pipeline, args.seed)
    
    train_loader, test_loader = data_loaders(args.dataset, TRAIN_DATA, TEST_DATA, batch_size)
    
    teacher, student = load_models(args.dataset, device, num_classes, TEACHER_PATH)
    
    train(device, learning_rate, batch_size, temperature, weight_decay, head_group, mid_group, tail_group, train_loader, test_loader, teacher, student, forget_class, num_epochs, num_classes, args.pipeline, STUDENT_SAVE_DIR, P_AVG_PLOT_DIR, P_K_BAR_PLOT_DIR)



if __name__ == "__main__":
    main()

   
