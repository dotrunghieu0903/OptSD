#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pruning Analysis Script

This script runs experiments with different pruning rates and collects accuracy
metrics to generate a graph showing the relationship between pruning rate and model performance.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from pruned.py
from pruned import (
    load_model, apply_pruning_method, create_pipeline,
    load_coco_captions, generate_image_and_monitor
)

from shared.cleanup import setup_memory_optimizations_pruning
from shared.metrics import calculate_clip_score, calculate_lpips, calculate_fid_subset

def run_pruning_experiment(pruning_rates, method="magnitude", num_images=50, dataset="coco", structured_dim=0):
    """
    Run pruning experiments with different pruning rates and collect accuracy metrics.
    
    Args:
        pruning_rates: List of pruning rates to test
        method: Pruning method to use
        num_images: Number of images to test per pruning rate
        dataset: Dataset to use (coco or flickr8k)
        structured_dim: Dimension for structured pruning
        
    Returns:
        Dict with results for each pruning rate
    """
    # Setup memory optimizations
    setup_memory_optimizations_pruning()
    
    results = {
        "pruning_rates": pruning_rates,
        "clip_scores": [],
        "lpips_scores": [],
        "fid_scores": [],
        "generation_times": []
    }
    
    # Create base output directory
    base_output_dir = os.path.join("pruning", "analysis_output")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load the dataset
    if dataset == "flickr8k":
        from dataset.flickr8k import process_flickr8k
        image_filename_to_caption, image_dimensions = process_flickr8k(limit=num_images)
        original_dir = "flickr8k/Images"
    else:  # coco
        annotations_file = "coco/annotations/captions_val2017.json"
        image_filename_to_caption, image_dimensions = load_coco_captions(annotations_file, limit=num_images)
        original_dir = "coco/val2017"
    
    # Use a subset of images for faster processing
    filenames_captions = list(image_filename_to_caption.items())[:num_images]
    
    # Test each pruning rate
    for pruning_rate in tqdm(pruning_rates, desc="Testing pruning rates"):
        print(f"\n=== Testing Pruning Rate: {pruning_rate} ===")
        
        # Create output directory for this pruning rate
        rate_output_dir = os.path.join(base_output_dir, f"{method}_rate_{pruning_rate}")
        os.makedirs(rate_output_dir, exist_ok=True)
        
        # Subdirectories for generated and original images
        generated_dir = os.path.join(rate_output_dir, "generated")
        os.makedirs(generated_dir, exist_ok=True)
        
        resized_original_dir = os.path.join(rate_output_dir, "resized_original")
        os.makedirs(resized_original_dir, exist_ok=True)
        
        # 1. Load the model
        transformer = load_model()
        
        # 2. Apply pruning
        pruned_transformer, pruning_stats = apply_pruning_method(
            transformer,
            method=method,
            amount=pruning_rate,
            structured_dim=structured_dim
        )
        
        # 3. Create pipeline with pruned model
        try:
            pipeline = create_pipeline(pruned_transformer)
            
            # 4. Generate images
            generation_times = []
            generated_images = []
            original_image_paths = []
            
            for i, (filename, caption) in enumerate(tqdm(filenames_captions, desc="Generating images")):
                try:
                    # Generate the image with resource monitoring
                    image, gen_time, _ = generate_image_and_monitor(
                        pipeline, caption, seed=i
                    )
                    
                    # Save the generated image
                    image_path = os.path.join(generated_dir, f"generated_{i}.png")
                    image.save(image_path)
                    generated_images.append(image_path)
                    
                    # Keep track of the original image
                    original_path = os.path.join(original_dir, filename)
                    original_image_paths.append(original_path)
                    
                    generation_times.append(gen_time)
                    
                except Exception as e:
                    print(f"Error generating image {i}: {e}")
            
            # 5. Calculate metrics
            # Calculate CLIP score
            clip_score = calculate_clip_score(generated_images, [caption for _, caption in filenames_captions])
            
            # Calculate LPIPS score
            lpips_score = calculate_lpips(generated_images, original_image_paths, resized_original_dir)
            
            # Calculate FID score
            fid_score = calculate_fid_subset(generated_images, original_image_paths)
            
            # Store the results
            results["clip_scores"].append(clip_score)
            results["lpips_scores"].append(lpips_score)
            results["fid_scores"].append(fid_score)
            results["generation_times"].append(np.mean(generation_times))
            
            # Clean up GPU memory
            del pipeline
            del pruned_transformer
            del transformer
            import torch
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error with pruning rate {pruning_rate}: {e}")
            # If an error occurs, append None values
            results["clip_scores"].append(None)
            results["lpips_scores"].append(None)
            results["fid_scores"].append(None)
            results["generation_times"].append(None)
    
    # Save the results
    results_file = os.path.join(base_output_dir, f"pruning_analysis_results_{method}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_results(results, output_dir="pruning/analysis_output"):
    """
    Create plots from the experiment results.
    
    Args:
        results: Dict with experiment results
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for accuracy metrics
    plt.figure(figsize=(12, 8))
    
    # CLIP score is our main accuracy metric (higher is better)
    if any(score is not None for score in results["clip_scores"]):
        plt.plot(
            results["pruning_rates"], 
            results["clip_scores"], 
            'o-', 
            label="CLIP Score (higher is better)"
        )
    
    # FID score (lower is better, so we invert it for the plot)
    if any(score is not None for score in results["fid_scores"]):
        max_fid = max([s for s in results["fid_scores"] if s is not None]) * 1.2
        normalized_fid = [max_fid - s if s is not None else None for s in results["fid_scores"]]
        plt.plot(
            results["pruning_rates"], 
            normalized_fid, 
            's-', 
            label="Inverted FID Score (higher is better)"
        )
    
    # Add LPIPS as a secondary metric (lower is better, so we invert it)
    if any(score is not None for score in results["lpips_scores"]):
        max_lpips = max([s for s in results["lpips_scores"] if s is not None]) * 1.2
        normalized_lpips = [max_lpips - s if s is not None else None for s in results["lpips_scores"]]
        plt.plot(
            results["pruning_rates"], 
            normalized_lpips, 
            '^-', 
            label="Inverted LPIPS (higher is better)"
        )
    
    plt.title("Accuracy vs. Pruning Rate")
    plt.xlabel("Pruning Rate")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "accuracy_vs_pruning_rate.png"), dpi=300)
    
    # Create figure for generation time
    plt.figure(figsize=(12, 8))
    plt.plot(results["pruning_rates"], results["generation_times"], 'o-', color='green')
    plt.title("Generation Time vs. Pruning Rate")
    plt.xlabel("Pruning Rate")
    plt.ylabel("Generation Time (seconds)")
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "time_vs_pruning_rate.png"), dpi=300)
    
    plt.close('all')
    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run pruning analysis experiments")
    parser.add_argument("--method", type=str, default="magnitude", 
                        choices=["magnitude", "structured", "iterative", "attention_heads"],
                        help="Pruning method to use")
    parser.add_argument("--num_images", type=int, default=50,
                        help="Number of images to test per pruning rate")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=["coco", "flickr8k"],
                        help="Dataset to use for testing")
    parser.add_argument("--structured_dim", type=int, default=0,
                        help="Dimension for structured pruning (0 for rows, 1 for columns)")
    
    args = parser.parse_args()
    
    # Define pruning rates to test
    pruning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Run the experiments
    results = run_pruning_experiment(
        pruning_rates=pruning_rates,
        method=args.method,
        num_images=args.num_images,
        dataset=args.dataset,
        structured_dim=args.structured_dim
    )
    
    # Plot the results
    plot_results(results)
    
    # Print the results
    print("\nExperiment Results:")
    print("===================")
    print("Pruning Rate | CLIP Score | FID Score | Inverted LPIPS | Generation Time (s)")
    print("-" * 75)
    
    for i, rate in enumerate(results["pruning_rates"]):
        clip = results["clip_scores"][i]
        fid = results["fid_scores"][i]
        lpips = results["lpips_scores"][i]
        time = results["generation_times"][i]
        
        # Convert CLIP Score to percentage (multiply by 100)
        clip_str = f"{clip*100:.2f}%" if clip is not None else "N/A"
        fid_str = f"{fid:.2f}" if fid is not None else "N/A"
        # Convert LPIPS to inverted percentage (1-lpips)*100 since lower LPIPS is better
        lpips_str = f"{(1-lpips)*100:.2f}%" if lpips is not None else "N/A"
        time_str = f"{time:.2f}" if time is not None else "N/A"
        
        print(f"{rate:^11} | {clip_str:^10} | {fid_str:^9} | {lpips_str:^14} | {time_str:^19}")

if __name__ == "__main__":
    main()