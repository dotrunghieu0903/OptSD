#!/bin/bash
# Run pruning analysis script with different methods

# Check if a method argument was provided
if [ -z "$1" ]; then
  METHOD="magnitude"
else
  METHOD="$1"
fi

# Check if number of images argument was provided
if [ -z "$2" ]; then
  NUM_IMAGES=50
else
  NUM_IMAGES="$2"
fi

# Check if dataset argument was provided
if [ -z "$3" ]; then
  DATASET="coco"
else
  DATASET="$3"
fi

echo "Running pruning analysis with method: $METHOD, dataset: $DATASET, using $NUM_IMAGES images per pruning rate"
python pruning/pruning_analysis.py --method "$METHOD" --num_images "$NUM_IMAGES" --dataset "$DATASET"

echo "Analysis complete. Check pruning/analysis_output/ for results."