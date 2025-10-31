#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Synthetic Accuracy vs. Pruning Rate Graph

This script creates a synthetic graph showing the relationship between pruning rate
and model accuracy to visualize the expected behavior without running experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

def generate_synthetic_data(pruning_rates, noise_level=0.05):
    """
    Generate synthetic accuracy data for visualization.
    
    Args:
        pruning_rates: List of pruning rates
        noise_level: Amount of noise to add to the data
        
    Returns:
        Dict with synthetic results
    """
    # Create synthetic clip scores (starting high, gradually declining,
    # then sharply dropping at higher pruning rates)
    clip_scores = []
    for rate in pruning_rates:
        if rate <= 0.4:
            # Slight decline up to 40% pruning
            score = 0.95 - (rate * 0.25)
        elif rate <= 0.7:
            # Steeper decline between 40% and 70% pruning
            score = 0.85 - ((rate - 0.4) * 0.8)
        else:
            # Sharp drop after 70% pruning
            score = 0.61 - ((rate - 0.7) * 1.5)
            
        # Add some noise
        noise = np.random.uniform(-noise_level, noise_level)
        score += noise
        clip_scores.append(max(0.1, min(1.0, score)))  # Clip to reasonable range
    
    # Create synthetic FID scores (starting low, gradually increasing)
    fid_scores = []
    for rate in pruning_rates:
        if rate <= 0.4:
            # Slight increase up to 40% pruning
            score = 10 + (rate * 10)
        elif rate <= 0.7:
            # Steeper increase between 40% and 70% pruning
            score = 14 + ((rate - 0.4) * 30)
        else:
            # Sharp increase after 70% pruning
            score = 23 + ((rate - 0.7) * 60)
            
        # Add some noise
        noise = np.random.uniform(-noise_level * 10, noise_level * 10)
        score += noise
        fid_scores.append(max(5, score))  # Ensure positive values
    
    # Create synthetic LPIPS scores (similar pattern to FID)
    lpips_scores = []
    for rate in pruning_rates:
        if rate <= 0.4:
            # Slight increase up to 40% pruning
            score = 0.15 + (rate * 0.1)
        elif rate <= 0.7:
            # Steeper increase between 40% and 70% pruning
            score = 0.19 + ((rate - 0.4) * 0.3)
        else:
            # Sharp increase after 70% pruning
            score = 0.28 + ((rate - 0.7) * 0.5)
            
        # Add some noise
        noise = np.random.uniform(-noise_level, noise_level)
        score += noise
        lpips_scores.append(max(0.05, min(0.8, score)))  # Clip to reasonable range
    
    # Create synthetic generation times (generally decreasing with pruning)
    generation_times = []
    for rate in pruning_rates:
        # Base time decreases as pruning increases
        time = 5.0 - (rate * 1.5)
        
        # Add some noise
        noise = np.random.uniform(-noise_level * 2, noise_level * 2)
        time += noise
        generation_times.append(max(1.0, time))  # Ensure positive values
    
    return {
        "pruning_rates": pruning_rates,
        "clip_scores": clip_scores,
        "fid_scores": fid_scores,
        "lpips_scores": lpips_scores,
        "generation_times": generation_times
    }

def plot_synthetic_results(results, output_dir="pruning/synthetic_results"):
    """
    Create plots from the synthetic results.
    
    Args:
        results: Dict with synthetic results
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for accuracy metrics
    plt.figure(figsize=(12, 8))
    
    # CLIP score (higher is better) - convert to percentage
    clip_percentages = [score * 100 for score in results["clip_scores"]]
    plt.plot(
        results["pruning_rates"], 
        clip_percentages, 
        'o-', 
        linewidth=2, 
        markersize=8,
        label="CLIP Score (higher is better)"
    )
    
    # FID score (lower is better, so we invert it for the plot)
    max_fid = max(results["fid_scores"]) * 1.2
    normalized_fid = [max_fid - s for s in results["fid_scores"]]
    plt.plot(
        results["pruning_rates"], 
        normalized_fid, 
        's-', 
        linewidth=2, 
        markersize=8,
        label="FID Score (higher is better)"
    )
    
    # Add LPIPS as a secondary metric - convert to inverted percentage
    inverted_lpips_percentages = [(1-s)*100 for s in results["lpips_scores"]]
    plt.plot(
        results["pruning_rates"], 
        inverted_lpips_percentages, 
        '^-', 
        linewidth=2, 
        markersize=8,
        label="LPIPS (higher is better)"
    )
    
    plt.title("Synthetic Accuracy vs. Pruning Rate")
    plt.xlabel("Pruning Rate")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    
    # Add pruning rates as x-tick labels
    plt.xticks(results["pruning_rates"])
    
    # Add annotations with the values
    for i, score in enumerate(results["clip_scores"]):
        plt.annotate(
            f"{score*100:.1f}%", 
            (results["pruning_rates"][i], score*100),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "synthetic_accuracy_vs_pruning_rate.png"), dpi=300, bbox_inches='tight')
    
    # Create figure for generation time
    plt.figure(figsize=(12, 8))
    plt.plot(
        results["pruning_rates"], 
        results["generation_times"], 
        'o-', 
        color='green',
        linewidth=2, 
        markersize=8
    )
    plt.title("Synthetic Generation Time vs. Pruning Rate")
    plt.xlabel("Pruning Rate")
    plt.ylabel("Generation Time (seconds)")
    plt.grid(True)
    
    # Add pruning rates as x-tick labels
    plt.xticks(results["pruning_rates"])
    
    # Add annotations with the values
    for i, time in enumerate(results["generation_times"]):
        plt.annotate(
            f"{time:.2f}s", 
            (results["pruning_rates"][i], time),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "synthetic_time_vs_pruning_rate.png"), dpi=300, bbox_inches='tight')
    
    plt.close('all')
    print(f"Synthetic plots saved to {output_dir}")
    
    # Save the data for later use
    with open(os.path.join(output_dir, "synthetic_results.json"), "w") as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pruning vs. accuracy graph")
    parser.add_argument("--output_dir", type=str, default="pruning/synthetic_results",
                        help="Directory to save the synthetic results and plots")
    parser.add_argument("--noise", type=float, default=0.05,
                        help="Noise level to add to synthetic data (0.0 to 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Define pruning rates to visualize
    pruning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Generate synthetic data
    results = generate_synthetic_data(pruning_rates, noise_level=args.noise)
    
    # Plot the results
    plot_synthetic_results(results, output_dir=args.output_dir)
    
    # Print the data in a table format
    print("\nSynthetic Results:")
    print("=================")
    print("Pruning Rate | CLIP Score | FID Score | LPIPS | Generation Time (s)")
    print("-" * 75)
    
    for i, rate in enumerate(results["pruning_rates"]):
        clip = results["clip_scores"][i]
        fid = results["fid_scores"][i]
        lpips = results["lpips_scores"][i]
        time = results["generation_times"][i]
        
        # Display CLIP as percentage and LPIPS as inverted percentage
        clip_percent = clip * 100
        lpips_percent = (1 - lpips) * 100
        
        print(f"{rate:^11} | {clip_percent:>6.2f}% | {fid:>7.2f} | {lpips_percent:>6.2f}% | {time:>15.2f}")

if __name__ == "__main__":
    main()