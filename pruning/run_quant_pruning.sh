#!/bin/bash

# Script to run the combined quantization and pruning script with COCO dataset

# Navigate to the OptSD directory
cd "$(dirname "$0")/.."

# Create output directory
mkdir -p combined_quant_pruning_outputs

echo "Running combined quantization and pruning with COCO dataset..."

# Clean up any existing output directory
rm -rf combined_quant_pruning_outputs/*

# Run with reduced steps for faster processing
python pruning/quant_pruning_coco.py \
  --pruning_amount 0.3 \
  --num_images 10 \
  --steps 25 \
  --guidance_scale 3.5 \
  --metrics_subset 5 \
  # Remove the comment below to skip metrics calculation if needed
  # --skip_metrics

echo "Process complete. Check 'combined_quant_pruning_outputs' for results."
