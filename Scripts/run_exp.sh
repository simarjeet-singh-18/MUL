#!/bin/bash

set -e

PYTHON_SCRIPT="/export/home/achyut/Simarjeet/MUL/Scripts/dtd.py"
OUTPUT_DIR="MUL/outputs_dtd_cifar10_cifar100_im100"

mkdir -p "$OUTPUT_DIR"

runs=(
    "--forget-class 0 --dataset "cifar10" --imb-factor 10 --seed 18"
    "--forget-class 3 --dataset "cifar10" --imb-factor 10 --seed 18"
    "--forget-class 8 --dataset "cifar10" --imb-factor 10 --seed 18"
    "--forget-class 0 --dataset "cifar100" --imb-factor 10 --seed 18"
    "--forget-class 25 --dataset "cifar100" --imb-factor 10 --seed 18"
    "--forget-class 64 --dataset "cifar100" --imb-factor 10 --seed 18"

    "--forget-class 0 --dataset "cifar10" --imb-factor 100 --seed 18"
    "--forget-class 3 --dataset "cifar10" --imb-factor 100 --seed 18"
    "--forget-class 8 --dataset "cifar10" --imb-factor 100 --seed 18"
    "--forget-class 0 --dataset "cifar100" --imb-factor 100 --seed 18"
    "--forget-class 25 --dataset "cifar100" --imb-factor 100 --seed 18"
    "--forget-class 64 --dataset "cifar100" --imb-factor 100 --seed 18"
    
    "--forget-class 0 --dataset "cifar10" --imb-factor 200 --seed 18"
    "--forget-class 3 --dataset "cifar10" --imb-factor 200 --seed 18"
    "--forget-class 8 --dataset "cifar10" --imb-factor 200 --seed 18"
    "--forget-class 0 --dataset "cifar100" --imb-factor 200 --seed 18"
    "--forget-class 25 --dataset "cifar100" --imb-factor 200 --seed 18"
    "--forget-class 64 --dataset "cifar100" --imb-factor 200 --seed 18"
)

for i in "${!runs[@]}"; do
    run_number=$((i + 1))
    output_file="$OUTPUT_DIR/run_${run_number}.out"

    echo "========================================"
    echo "Running experiment $run_number"
    echo "Arguments: ${runs[$i]}"
    echo "Output: $output_file"
    echo "========================================"

    python3 -u "$PYTHON_SCRIPT" ${runs[$i]} > "$output_file" 2>&1

    echo "Experiment $run_number completed."
done

echo "All experiments completed successfully."