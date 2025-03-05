#!/bin/bash

# Default values (can be overridden by command-line arguments)
TASK_NAME="ocr"
NUM_SAMPLES=3000

# Define the dataset splits
splits=("train" "validation" "test")

for split in "${splits[@]}"; do
    echo "Running model: deepseek-vl2-tiny on split: $split"
    /home/hice1/stekin6/.conda/envs/deepseek/bin/python inference_deepseek_open_ended.py \
    --model_name "deepseek-vl2-tiny" \
    --task_name "$TASK_NAME" \
    --dataset_type "$split" \
    --num_samples "$NUM_SAMPLES"
done


for split in "${splits[@]}"; do
    echo "Running model: deepseek-vl2-small on split: $split"
    /home/hice1/stekin6/.conda/envs/deepseek/bin/python inference_deepseek_open_ended.py \
    --model_name "deepseek-vl2-small" \
    --task_name "$TASK_NAME" \
    --dataset_type "$split" \
    --num_samples "$NUM_SAMPLES"
done
