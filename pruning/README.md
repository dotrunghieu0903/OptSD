# Pruning Analysis for Diffusion Models

This directory contains tools for analyzing the relationship between pruning rate and model performance for diffusion models. The scripts allow you to run experiments with different pruning rates and methods, collect accuracy metrics, and visualize the results.

## Available Scripts

### 1. Full Pruning Analysis

Run experiments with multiple pruning rates (0.1 to 0.9) and collect accuracy metrics:

```bash
./run_pruning_analysis.sh [method] [num_images] [dataset]
```

- `method`: Pruning method to use (magnitude, structured, iterative, attention_heads)
- `num_images`: Number of images to test per pruning rate
- `dataset`: Dataset to use for testing (coco, flickr8k)

Example:
```bash
./run_pruning_analysis.sh magnitude 50 coco
```

### 2. Test Individual Pruning Rate

Test a specific pruning rate to quickly collect a data point:

```bash
python pruning/test_single_rate.py --rate [rate] --method [method] --num_images [num_images] --dataset [dataset]
```

Example:
```bash
python pruning/test_single_rate.py --rate 0.3 --method magnitude --num_images 10 --dataset coco
```

### 3. Visualize Results

Visualize the results of the experiments:

```bash
python pruning/visualize_results.py [--results_file path/to/results.json] [--methods method1 method2 ...] [--output_dir output/dir] [--compare]
```

Example:
```bash
python pruning/visualize_results.py --methods magnitude structured --compare
```

### 4. Generate Synthetic Graph

Generate a synthetic graph showing the expected relationship between pruning rate and accuracy:

```bash
python pruning/generate_synthetic_graph.py [--output_dir output/dir] [--noise 0.05] [--seed 42]
```

## Understanding the Results

The analysis collects several metrics for each pruning rate:

1. **CLIP Score**: Measures semantic similarity between generated images and their captions (higher is better)
2. **FID Score**: Measures the quality and diversity of generated images (lower is better)
3. **LPIPS Score**: Measures perceptual similarity between generated and original images (lower is better)
4. **Generation Time**: Average time to generate an image (lower is better)

The main graph shows the relationship between pruning rate (x-axis) and accuracy (y-axis). For visualization purposes, FID and LPIPS scores are inverted so that higher values on the graph represent better performance for all metrics.

## Example Output

The analysis produces several files:

1. JSON files with raw results for each pruning method
2. PNG images with graphs showing the relationship between pruning rate and model performance
3. Text files with summaries of the results

## Requirements

- Python 3.8+
- PyTorch
- matplotlib
- numpy
- tqdm
- PIL

## Adding New Metrics

To add new metrics to the analysis, modify the `run_pruning_experiment` function in `pruning_analysis.py` to collect and store the additional metrics.