#!/bin/bash

# Run advanced pruning script
# This script applies pruning to a diffusion model and evaluates on COCO dataset

echo "Starting advanced pruning with dataset..."

# Default values
PRUNING_METHOD="magnitude"  # Options: magnitude, structured, iterative, attention_heads
PRUNING_AMOUNT=0.3          # Amount to prune (0.0 to 0.9)
NUM_IMAGES=500              # Number of images to generate
INFERENCE_STEPS=50          # Number of inference steps
GUIDANCE_SCALE=3.5          # Guidance scale for generation
METRICS_SUBSET=50           # Number of images for metrics calculation
USE_FLICKR8K=True           # Use Flickr8k dataset if set
USE_COCO=True               # Use COCO dataset by default

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --method)
      PRUNING_METHOD="$2"
      shift 2
      ;;
    --amount)
      PRUNING_AMOUNT="$2"
      shift 2
      ;;
    --images)
      NUM_IMAGES="$2"
      shift 2
      ;;
    --steps)
      INFERENCE_STEPS="$2"
      shift 2
      ;;
    --guidance)
      GUIDANCE_SCALE="$2"
      shift 2
      ;;
    --metrics_subset)
      METRICS_SUBSET="$2"
      shift 2
      ;;
    --use_flickr8k)
      USE_FLICKR8K=True
      shift
      ;;
    --skip_metrics)
      SKIP_METRICS="--skip_metrics"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --method METHOD       Pruning method (magnitude, structured, iterative, attention_heads)"
      echo "  --amount AMOUNT       Pruning amount (0.0 to 0.9)"
      echo "  --images NUM          Number of images to generate"
      echo "  --steps STEPS         Number of inference steps"
      echo "  --guidance SCALE      Guidance scale for generation"
      echo "  --metrics_subset NUM  Number of images for metrics calculation"
      echo "  --skip_metrics        Skip calculation of image quality metrics"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d")
OUTPUT_DIR="pruning/${PRUNING_METHOD}_pruned_outputs_${TIMESTAMP}"

echo "Configuration:"
echo "  Pruning method: $PRUNING_METHOD"
echo "  Pruning amount: $PRUNING_AMOUNT"
echo "  Number of images: $NUM_IMAGES"
echo "  Inference steps: $INFERENCE_STEPS"
echo "  Guidance scale: $GUIDANCE_SCALE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Metrics subset: $METRICS_SUBSET"
echo ""


# Build the command
CMD="python pruning/advanced_pruning.py \
  --pruning_method \"$PRUNING_METHOD\" \
  --pruning_amount \"$PRUNING_AMOUNT\" \
  --num_images \"$NUM_IMAGES\" \
  --steps \"$INFERENCE_STEPS\" \
  --guidance_scale \"$GUIDANCE_SCALE\" \
  --metrics_subset \"$METRICS_SUBSET\" \
  --output_dir \"$OUTPUT_DIR\""

# Add --use_flickr8k flag if enabled
if [ "$USE_FLICKR8K" = "True" ]; then
  CMD="$CMD --use_flickr8k"
fi

# Add skip metrics if set
if [ -n "$SKIP_METRICS" ]; then
  CMD="$CMD $SKIP_METRICS"
fi

# Run the command
eval $CMD

# Check if the script executed successfully
if [ $? -eq 0 ]; then
  echo "Advanced pruning completed successfully!"
  echo "Results saved to $OUTPUT_DIR"
else
  echo "Advanced pruning failed. Please check the logs for errors."
  exit 1
fi

# ./pruning/run_advanced_pruning.sh