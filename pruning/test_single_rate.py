#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Individual Pruning Rate Tester

This script runs a single experiment with a specific pruning rate to quickly
collect data points for the accuracy vs. pruning rate curve.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
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

def test_pruning_rate(pruning_rate, method="magnitude", num_images=10, dataset="coco", structured_dim=0):
    """
    Test a specific pruning rate and collect accuracy metrics.
    
    Args:
        pruning_rate: The pruning rate to test
        method: Pruning method to use
        num_images: Number of images to test
        dataset: Dataset to use (coco or flickr8k)
        structured_dim: Dimension for structured pruning
        
    Returns:
        Dict with results for this pruning rate
    """
    # Setup memory optimizations
    setup_memory_optimizations_pruning()
    
    results = {
        "pruning_rate": pruning_rate,
        "method": method,
        "clip_score": None,
        "lpips_score": None,
        "fid_score": None,
        "generation_time": None
    }
    
    # Create output directory
    base_output_dir = os.path.join("pruning", "analysis_output", f"{method}_rate_{pruning_rate}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Subdirectories for generated and original images
    generated_dir = os.path.join(base_output_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)
    
    resized_original_dir = os.path.join(base_output_dir, "resized_original")
    os.makedirs(resized_original_dir, exist_ok=True)
    
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
    
    print(f"\n=== Testing Pruning Rate: {pruning_rate} ===")
    
    try:
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
        
        if generated_images:
            # 5. Calculate metrics
            # Calculate CLIP score
            clip_score = calculate_clip_score(generated_images, [caption for _, caption in filenames_captions])
            results["clip_score"] = clip_score
            
            # Calculate LPIPS score
            lpips_score = calculate_lpips(generated_images, original_image_paths, resized_original_dir)
            results["lpips_score"] = lpips_score
            
            # Calculate FID score
            fid_score = calculate_fid_subset(generated_images, original_image_paths)
            results["fid_score"] = fid_score
            
            # Store the generation time
            results["generation_time"] = np.mean(generation_times)
            
            print(f"\nResults for pruning rate {pruning_rate}:")
            print(f"CLIP Score: {clip_score*100:.2f}%")
            print(f"Inverted LPIPS Score: {(1-lpips_score)*100:.2f}% (higher is better)")
            print(f"FID Score: {fid_score:.2f} (lower is better)")
            print(f"Average Generation Time: {results['generation_time']:.2f} seconds")
        
        # Clean up GPU memory
        del pipeline
        del pruned_transformer
        del transformer
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"Error with pruning rate {pruning_rate}: {e}")
    
    # Save the results
    results_file = os.path.join(base_output_dir, f"single_test_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test an individual pruning rate")
    parser.add_argument("--rate", type=float, required=True,
                        help="Pruning rate to test (between 0.1 and 0.9)")
    parser.add_argument("--method", type=str, default="magnitude", 
                        choices=["magnitude", "structured", "iterative", "attention_heads"],
                        help="Pruning method to use")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of images to test")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=["coco", "flickr8k"],
                        help="Dataset to use for testing")
    parser.add_argument("--structured_dim", type=int, default=0,
                        help="Dimension for structured pruning (0 for rows, 1 for columns)")
    
    args = parser.parse_args()
    
    # Validate pruning rate
    if args.rate < 0.1 or args.rate > 0.9:
        print("Error: Pruning rate must be between 0.1 and 0.9")
        return 1
    
    # Run the test
    results = test_pruning_rate(
        pruning_rate=args.rate,
        method=args.method,
        num_images=args.num_images,
        dataset=args.dataset,
        structured_dim=args.structured_dim
    )
    
    return 0

if __name__ == "__main__":
    main()