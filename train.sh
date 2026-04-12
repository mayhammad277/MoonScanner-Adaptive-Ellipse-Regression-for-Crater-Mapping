#!/bin/bash
set -e

DATA_DIR=$1

echo "===================================="
echo "TRAIN SCRIPT STARTED"
echo "Training data location: ${DATA_DIR}"
echo "===================================="

# TopCoder rule: remove shipped model
echo "Removing existing model file (if any)"
rm -f swin_crater_best_v8_d5.pth

echo "Starting training..."
python3 train.py \
    --data_dir "${DATA_DIR}" \
    --gt_csv "${DATA_DIR}/train-gt.csv" \
    --output_model "swin_crater_best_v8_d5.pth"

echo "===================================="
echo "TRAIN SCRIPT FINISHED"
echo "===================================="

