#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pruning Results Visualization Script

This script visualizes the results of the pruning analysis experiments,
creating plots showing the relationship between pruning rate and model performance.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_results(results_file):
    """
    Load results from a JSON file.
    
    Args:
        results_file: Path to the JSON file with results
        
    Returns:
        Dict with experiment results
    """
    with open(results_file, "r") as f:
        return json.load(f)

def plot_results(results, output_dir="pruning/analysis_output", title_prefix="", save_suffix=""):
    """
    Create plots from the experiment results.
    
    Args:
        results: Dict with experiment results
        output_dir: Directory to save the plots
        title_prefix: Prefix for plot titles
        save_suffix: Suffix for saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for accuracy metrics
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Get the pruning rates
    pruning_rates = results["pruning_rates"]
    
    # CLIP score is our main accuracy metric (higher is better)
    if "clip_scores" in results and any(score is not None for score in results["clip_scores"]):
        # Convert CLIP scores to percentages for display
        clip_percentages = [score * 100 if score is not None else None for score in results["clip_scores"]]
        line1 = ax1.plot(
            pruning_rates, 
            clip_percentages, 
            'o-', 
            linewidth=2, 
            markersize=8,
            label="CLIP Score (higher is better)",
            color='blue'
        )
    
    # Add LPIPS as a secondary metric (lower is better, so we invert it)
    if "lpips_scores" in results and any(score is not None for score in results["lpips_scores"]):
        # Convert LPIPS to inverted percentages: (1-lpips)*100
        inverted_lpips_percentages = [(1-s)*100 if s is not None else None for s in results["lpips_scores"]]
        line2 = ax1.plot(
            pruning_rates, 
            inverted_lpips_percentages, 
            '^-', 
            linewidth=2, 
            markersize=8,
            label="LPIPS % (higher is better)",
            color='green'
        )
    
    ax1.set_xlabel("Pruning Rate")
    ax1.set_ylabel("Score (%)")
    ax1.grid(True)
    ax1.set_xticks(pruning_rates)
    
    # Create secondary y-axis for FID score (lower is better)
    ax2 = ax1.twinx()
    
    # FID score (lower is better)
    if "fid_scores" in results and any(score is not None for score in results["fid_scores"]):
        line3 = ax2.plot(
            pruning_rates, 
            results["fid_scores"], 
            's-', 
            linewidth=2, 
            markersize=8,
            label="FID Score (lower is better)",
            color='red'
        )
        ax2.set_ylabel("FID Score (lower is better)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(f"{title_prefix}Model Performance vs. Pruning Rate")
    
    # Add annotations with the values
    if "clip_scores" in results:
        for i, score in enumerate(results["clip_scores"]):
            if score is not None:
                ax1.annotate(
                    f"{score*100:.1f}%", 
                    (pruning_rates[i], score*100),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    color='blue'
                )
    
    # Add FID score annotations
    if "fid_scores" in results:
        for i, score in enumerate(results["fid_scores"]):
            if score is not None:
                ax2.annotate(
                    f"{score:.1f}", 
                    (pruning_rates[i], score),
                    textcoords="offset points",
                    xytext=(0,-15),
                    ha='center',
                    color='red'
                )
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f"accuracy_vs_pruning_rate{save_suffix}.png"), dpi=300, bbox_inches='tight')
    
    # Create figure for generation time
    if "generation_times" in results:
        plt.figure(figsize=(12, 8))
        plt.plot(
            pruning_rates, 
            results["generation_times"], 
            'o-', 
            color='green',
            linewidth=2, 
            markersize=8
        )
        plt.title(f"{title_prefix}Generation Time vs. Pruning Rate")
        plt.xlabel("Pruning Rate")
        plt.ylabel("Generation Time (seconds)")
        plt.grid(True)
        
        # Add pruning rates as x-tick labels
        plt.xticks(pruning_rates)
        
        # Add annotations with the values
        for i, time in enumerate(results["generation_times"]):
            if time is not None:
                plt.annotate(
                    f"{time:.2f}s", 
                    (pruning_rates[i], time),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center'
                )
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f"time_vs_pruning_rate{save_suffix}.png"), dpi=300, bbox_inches='tight')
    
    # Create separate figure for FID score (lower is better)
    if "fid_scores" in results and any(score is not None for score in results["fid_scores"]):
        plt.figure(figsize=(12, 8))
        plt.plot(
            pruning_rates, 
            results["fid_scores"], 
            's-', 
            color='red',
            linewidth=2, 
            markersize=8
        )
        plt.title(f"{title_prefix}FID Score vs. Pruning Rate (Lower is Better)")
        plt.xlabel("Pruning Rate")
        plt.ylabel("FID Score")
        plt.grid(True)
        
        # Add pruning rates as x-tick labels
        plt.xticks(pruning_rates)
        
        # Add annotations with the values
        for i, score in enumerate(results["fid_scores"]):
            if score is not None:
                plt.annotate(
                    f"{score:.2f}", 
                    (pruning_rates[i], score),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center'
                )
        
        # Add text indicating lower is better
        plt.text(0.02, 0.98, "Lower FID = Better Quality", 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f"fid_vs_pruning_rate{save_suffix}.png"), dpi=300, bbox_inches='tight')
    
    plt.close('all')
    print(f"Plots saved to {output_dir}")

def create_comparison_plot(results_files, methods, output_dir="pruning/analysis_output"):
    """
    Create a comparison plot from multiple experiment results.
    
    Args:
        results_files: List of paths to results files
        methods: List of method names corresponding to results_files
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for accuracy comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Define colors and markers for different methods
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'x']
    
    # Plot 1: CLIP Score comparison (higher is better)
    for i, (results_file, method) in enumerate(zip(results_files, methods)):
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
            
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Plot CLIP score as main accuracy metric
            if "clip_scores" in results and any(score is not None for score in results["clip_scores"]):
                # Convert to percentages
                clip_percentages = [score * 100 if score is not None else None for score in results["clip_scores"]]
                ax1.plot(
                    results["pruning_rates"], 
                    clip_percentages, 
                    marker=marker, 
                    color=color,
                    linewidth=2, 
                    markersize=8,
                    linestyle='-',
                    label=f"{method}"
                )
        except Exception as e:
            print(f"Error loading results from {results_file}: {e}")
    
    ax1.set_title("CLIP Score Comparison (Higher is Better)")
    ax1.set_xlabel("Pruning Rate")
    ax1.set_ylabel("CLIP Score (%)")
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: FID Score comparison (lower is better)
    for i, (results_file, method) in enumerate(zip(results_files, methods)):
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
            
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Plot FID score
            if "fid_scores" in results and any(score is not None for score in results["fid_scores"]):
                ax2.plot(
                    results["pruning_rates"], 
                    results["fid_scores"], 
                    marker=marker, 
                    color=color,
                    linewidth=2, 
                    markersize=8,
                    linestyle='-',
                    label=f"{method}"
                )
        except Exception as e:
            print(f"Error loading results from {results_file}: {e}")
    
    ax2.set_title("FID Score Comparison (Lower is Better)")
    ax2.set_xlabel("Pruning Rate")
    ax2.set_ylabel("FID Score")
    ax2.grid(True)
    ax2.legend()
    
    # Add text indicating lower is better for FID
    ax2.text(0.02, 0.98, "Lower FID = Better Quality", 
            transform=ax2.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the comparison figure
    plt.savefig(os.path.join(output_dir, "pruning_methods_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize pruning analysis results")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Path to the JSON file with results (optional)")
    parser.add_argument("--methods", type=str, nargs='+', 
                        default=["magnitude", "structured", "iterative", "attention_heads"],
                        help="Methods to visualize (default: all)")
    parser.add_argument("--output_dir", type=str, default="pruning/analysis_output",
                        help="Directory to save the plots")
    parser.add_argument("--compare", action="store_true",
                        help="Create a comparison plot of multiple methods")
    
    args = parser.parse_args()
    
    # If a specific results file is provided, visualize it
    if args.results_file:
        try:
            results = load_results(args.results_file)
            method_name = os.path.basename(args.results_file).replace("pruning_analysis_results_", "").replace(".json", "")
            plot_results(
                results, 
                output_dir=args.output_dir, 
                title_prefix=f"{method_name.capitalize()} Pruning: ", 
                save_suffix=f"_{method_name}"
            )
        except Exception as e:
            print(f"Error visualizing results from {args.results_file}: {e}")
    else:
        # Otherwise, visualize all specified methods
        results_files = []
        valid_methods = []
        
        for method in args.methods:
            results_file = os.path.join(args.output_dir, f"pruning_analysis_results_{method}.json")
            if os.path.exists(results_file):
                try:
                    results = load_results(results_file)
                    plot_results(
                        results, 
                        output_dir=args.output_dir, 
                        title_prefix=f"{method.capitalize()} Pruning: ", 
                        save_suffix=f"_{method}"
                    )
                    results_files.append(results_file)
                    valid_methods.append(method)
                except Exception as e:
                    print(f"Error visualizing results for {method}: {e}")
            else:
                print(f"No results file found for method: {method}")
        
        # Create comparison plot if requested
        if args.compare and len(results_files) > 1:
            create_comparison_plot(results_files, valid_methods, output_dir=args.output_dir)

if __name__ == "__main__":
    main()