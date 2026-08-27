import time
import torch
import random
import argparse
import numpy as np
from neggrad import run_neggrad_unlearning



######################################################################

imb_factor = 1000
dataset = "cifar10" # cifar10 or cifar100 or food101 or places365
forget_class = "head" # head or mid or tail or numeric
clustering_type = "mean" # manual or mean or mvm
pipeline = "imp" # imp or nor

######################################################################



if dataset == "cifar10":
    TRAIN_DATA = f"/export/home/achyut/Simarjeet/MUL/Datasets/cifar10/Generated/cifar10_lt_im{imb_factor}.pt"
    TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/cifar10"
    NUM_CLASSES = 10
    NUM_EPOCHS = 15
elif dataset == "cifar100":
    TRAIN_DATA = f"/export/home/achyut/Simarjeet/MUL/Datasets/cifar100/Generated/cifar100_lt_im{imb_factor}.pt"
    TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/cifar100"
    NUM_CLASSES = 100
    NUM_EPOCHS = 30
elif dataset == "food101":
    TRAIN_DATA = f"/export/home/achyut/Simarjeet/MUL/Datasets/food101/food-101/split/train_im{imb_factor}"
    TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/food101/food-101/split/val"
    NUM_CLASSES = 101
    NUM_EPOCHS = 30
elif dataset == "places365":
    TRAIN_DATA = f"/export/home/achyut/Simarjeet/MUL/Datasets/places365/lt/train_im{imb_factor}"
    TEST_DATA = "/export/home/achyut/Simarjeet/MUL/Datasets/places365/lt/val"
    NUM_CLASSES = 365
    NUM_EPOCHS = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_root = f"/export/home/achyut/Simarjeet/Unlearning/KD-MUL-CIFAR100/Data/Generated/cifar100_lt_im{imb_factor}.pt"

start_time = time.perf_counter()

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



def parse_args():
    p = argparse.ArgumentParser(description="State of the art Machine Unlearning Methods")
    
    p.add_argument("--dataset", default="cifar10", help="cifar10 or cifar100 or food101 or places365")
    p.add_argument("--forget-class", default="head", help="head or mid or tail or numeric")
    p.add_argument("--clustering-type", default="manual", help="manual or mean or mvm")
    p.add_argument("--pipeline", default="imp", help="imp or nor")
    
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()



def main():
    args = parse_args()
    
    run_neggrad_unlearning(
        original_ckpt=f"/export/home/achyut/Simarjeet/Unlearning/KD-MUL-CIFAR100/Models/Teachers/teacher_im{imb_factor}.pth",
        save_ckpt_path=f"/export/home/achyut/Simarjeet/Unlearning/KD-MUL-CIFAR100/Models/SOA/Neggrad/{imb_factor}/neggrad_im2{imb_factor}_cls{forget_class}.pth",
        forget_class=forget_class,
        device=device,
        num_workers=0,
        root=data_root
    )

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Script execution time: {execution_time:.6f} seconds")   



if __name__ == "__main__":
    main()