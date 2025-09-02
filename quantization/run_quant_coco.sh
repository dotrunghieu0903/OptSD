#!/bin/bash

# Script to run the quantization script with COCO dataset

# Navigate to the OptSD directory
cd "$(dirname "$0")/.."

# Create output directory
mkdir -p quantization/quant_outputs

echo "Running quantization with COCO dataset..."

# Clean up any existing output directory
rm -rf quantization/quant_outputs/*

# Run with parameters
python quantization/quant_coco.py --num_images 5 --steps 50 --guidance_scale 3.5 --metrics_subset 5
  
# Add --skip_metrics to skip metrics calculation if needed
# python quantization/quant_coco.py --num_images 5 --steps 25 --guidance_scale 3.5 --metrics_subset 5 --skip_metrics

python kvcache/sam_kvcache.py --coco_dir /coco --num_images 5 --benchmark --num_runs 5 --metrics_subset 20