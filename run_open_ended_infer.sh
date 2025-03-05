#!/bin/bash

# Default values (can be overridden by command-line arguments)
TASK_NAME="ocr"
NUM_SAMPLES=3000

# Define the list of models
models=("llava-v1.6-vicuna-7b-hf" "llava-v1.6-vicuna-13b-hf" \
        "Qwen2.5-VL-7B-Instruct" "InternVL2-8B")

# Define the dataset splits
splits=("train" "validation" "test")

for model in "${models[@]}"; do
    for split in "${splits[@]}"; do
        echo "Running model: $model on split: $split"
        python inference_scripts/inference_open_ended.py \
        --model_name "$model" \
        --task_name "$TASK_NAME" \
        --dataset_type "$split" \
        --num_samples "$NUM_SAMPLES"
    done
done
