#!/bin/bash

# Default values (can be overridden by command-line arguments)
TASK_NAME="mmmu"
MODEL_NAME="InternVL2-8B"
DATASET_TYPE="validation"
NUM_SAMPLES=1500


python inference.py \
    --task_name "$TASK_NAME" \
    --model_name "$MODEL_NAME" \
    --dataset_type "$DATASET_TYPE" \
    --num_samples "$NUM_SAMPLES"


python inference.py \
    --task_name "okvqa" \
    --model_name "$MODEL_NAME" \
    --dataset_type "train" \
    --num_samples "$NUM_SAMPLES"


python inference.py \
    --task_name "okvqa" \
    --model_name "$MODEL_NAME" \
    --dataset_type $DATASET_TYPE \
    --num_samples "$NUM_SAMPLES"
