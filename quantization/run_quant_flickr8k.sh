#!/bin/bash

# Script to run the quantization script with Flickr30k dataset

# Navigate to the OptSD directory
cd "$(dirname "$0")/.."

# Clean up any existing output directory
rm -rf quantization/quant_outputs/flickr8k/*

# Create output directory
mkdir -p quantization/quant_outputs/flickr8k

echo "Running quantization with Flickr8k dataset..."

# Run with parameters
python quantization/quant_flickr8k.py --num_images 10 --steps 50 --guidance_scale 3.5 --metrics_subset 5

# Add --skip_metrics to skip metrics calculation if needed
# python quantization/quant_flickr30k.py --num_images 5 --steps 25 --guidance_scale 3.5 --metrics_subset 5 --skip_metrics
