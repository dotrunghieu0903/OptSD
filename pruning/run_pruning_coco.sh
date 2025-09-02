#!/bin/bash

# Run script for pruning evaluation on COCO dataset
# This script runs the pruning_coco.py script with configurable parameters

# Default values
PRUNING_AMOUNT=0.5
PRUNING_METHOD="magnitude"
NUM_IMAGES=5
STEPS=25
GUIDANCE_SCALE=7.0
METRICS_SUBSET=100
SKIP_METRICS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --pruning_amount)
      PRUNING_AMOUNT="$2"
      shift 2
      ;;
    --pruning_method)
      PRUNING_METHOD="$2"
      shift 2
      ;;
    --num_images)
      NUM_IMAGES="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --guidance_scale)
      GUIDANCE_SCALE="$2"
      shift 2
      ;;
    --metrics_subset)
      METRICS_SUBSET="$2"
      shift 2
      ;;
    --skip_metrics)
      SKIP_METRICS=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print configuration
echo "=== Pruning Configuration ==="
echo "Pruning Amount: $PRUNING_AMOUNT"
echo "Pruning Method: $PRUNING_METHOD"
echo "Number of Images: $NUM_IMAGES"
echo "Inference Steps: $STEPS"
echo "Guidance Scale: $GUIDANCE_SCALE"
echo "Metrics Subset Size: $METRICS_SUBSET"
echo "Skip Metrics: $SKIP_METRICS"
echo "=========================="

# Create command with all parameters
CMD="python3 pruning/pruning_coco.py --pruning_amount $PRUNING_AMOUNT --pruning_method $PRUNING_METHOD --num_images $NUM_IMAGES --steps $STEPS --guidance_scale $GUIDANCE_SCALE --metrics_subset $METRICS_SUBSET"

# Add skip_metrics flag if enabled
if [ "$SKIP_METRICS" = true ]; then
  CMD="$CMD --skip_metrics"
fi

# Run the pruning evaluation
echo "Running: $CMD"
echo "=========================="
eval $CMD

# Exit with the same code as the python script
exit $?

python ./pruning/pruning_coco.py  --pruning_amount 0.3 --pruning_method magnitude --num_images 1000 --steps 50 --guidance_scale 3.5 --metrics_subset 1000