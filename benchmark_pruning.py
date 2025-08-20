"""
Benchmarking script to compare performance of pruned vs unpruned diffusion models.
This script evaluates multiple pruning configurations and generates comparison metrics.
"""

import torch
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision
import os
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.utils.prune as prune
from metrics import calculate_clip_score, calculate_fid
import json

# Test prompts covering various image types
TEST_PROMPTS = [
    "A beautiful mountain landscape at sunset with a lake reflecting the sky",
    "A close-up portrait of a person with detailed facial features",
    "A futuristic cityscape with tall skyscrapers and flying vehicles",
    "A detailed painting of flowers in a vase with intricate textures",
    "An abstract digital artwork with vibrant colors and geometric shapes"
]

def get_model_size(model):
    """Calculate the size of the model in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def get_sparsity(model):
    """Calculate the sparsity of the model."""
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    sparsity = 100.0 * zero_params / total_params if total_params > 0 else 0
    return sparsity

def apply_pruning(model, method="magnitude", amount=0.3, structured=False):
    """Apply different pruning methods to the model."""
    print(f"Applying {method} pruning with amount {amount}...")
    
    if method == "magnitude":
        # Global unstructured magnitude pruning
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))
        
        if structured:
            # Structured pruning (removes entire rows/columns)
            for module, name in parameters_to_prune:
                prune.ln_structured(module, name=name, amount=amount, n=2, dim=0)
                prune.remove(module, name)
        else:
            # Apply global magnitude pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
            
            # Make pruning permanent
            for module, name in parameters_to_prune:
                prune.remove(module, name)
    
    elif method == "random":
        # Random pruning
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))
                
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=amount
        )
        
        # Make pruning permanent
        for module, name in parameters_to_prune:
            prune.remove(module, name)
    
    elif method == "layerwise":
        # Layer-wise pruning (different amounts for different layers)
        # Early layers get less pruning, later layers get more
        layer_count = 0
        total_layers = sum(1 for _ in model.named_modules() if isinstance(_, nn.Linear) or isinstance(_, nn.Conv2d))
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                # Increase pruning amount gradually for later layers
                layer_pruning = amount * (0.5 + 0.5 * layer_count / total_layers)
                layer_pruning = min(0.8, layer_pruning)  # Cap at 80% pruning
                
                prune.l1_unstructured(module, name='weight', amount=layer_pruning)
                prune.remove(module, 'weight')
                layer_count += 1
    
    sparsity = get_sparsity(model)
    print(f"Model sparsity after pruning: {sparsity:.2f}%")
    return model

def load_model(precision=None):
    """Load the diffusion model."""
    if precision is None:
        precision = get_precision()
        print(f"Detected precision: {precision}")
    
    model_path = f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(model_path)
    
    return transformer, precision

def create_pipeline(transformer):
    """Create a diffusion pipeline using the provided transformer."""
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    return pipeline

def generate_and_benchmark(pipeline, prompts, num_inference_steps=50, output_dir=None):
    """Generate images and benchmark performance."""
    results = {
        "times": [],
        "images": []
    }
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        # Warm up GPU
        if i == 0:
            _ = pipeline(prompt, num_inference_steps=10)
            torch.cuda.synchronize()
        
        # Measure generation time
        torch.cuda.synchronize()
        start_time = time.time()
        
        image = pipeline(prompt, num_inference_steps=num_inference_steps).images[0]
        
        torch.cuda.synchronize()
        end_time = time.time()
        generation_time = end_time - start_time
        
        results["times"].append(generation_time)
        results["images"].append(image)
        
        # Save the generated image if output directory is provided
        if output_dir:
            image_path = os.path.join(output_dir, f"image_{i}.png")
            image.save(image_path)
    
    # Calculate stats
    avg_time = sum(results["times"]) / len(results["times"])
    
    return results, avg_time

def main():
    parser = argparse.ArgumentParser(description="Benchmark pruned diffusion models")
    parser.add_argument("--precision", type=str, default=None,
                        help="Precision to use (fp4, fp8, etc.)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--output_dir", type=str, default="pruning_benchmark",
                        help="Directory to save benchmark results")
    parser.add_argument("--prompts", type=int, default=5,
                        help="Number of test prompts to use (max 5)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure test prompts
    test_prompts = TEST_PROMPTS[:min(args.prompts, len(TEST_PROMPTS))]
    
    # Configuration for different pruning methods to benchmark
    pruning_configs = [
        {"method": "magnitude", "amount": 0.3, "structured": False, "name": "magnitude_30"},
        {"method": "magnitude", "amount": 0.5, "structured": False, "name": "magnitude_50"},
        {"method": "magnitude", "amount": 0.3, "structured": True, "name": "structured_30"},
        {"method": "random", "amount": 0.3, "structured": False, "name": "random_30"},
        {"method": "layerwise", "amount": 0.3, "structured": False, "name": "layerwise_30"}
    ]
    
    # Load the original model
    print("\n=== Loading Original Model ===")
    transformer, precision = load_model(args.precision)
    original_size = get_model_size(transformer)
    print(f"Original model size: {original_size:.2f} MB")
    
    # Benchmark results storage
    results = {
        "original": {
            "size": original_size,
            "time": 0,
            "speedup": 1.0,
            "sparsity": get_sparsity(transformer)
        }
    }
    
    # First benchmark the original model
    pipeline = create_pipeline(transformer)
    print("\n=== Benchmarking Original Model ===")
    original_dir = os.path.join(args.output_dir, "original")
    os.makedirs(original_dir, exist_ok=True)
    
    original_results, original_time = generate_and_benchmark(
        pipeline, test_prompts, args.steps, original_dir
    )
    results["original"]["time"] = original_time
    print(f"Original model average generation time: {original_time:.2f}s")
    
    # Now benchmark each pruning configuration
    for config in pruning_configs:
        print(f"\n=== Testing {config['name']} Pruning ===")
        # Clone the model for this pruning configuration
        pruned_transformer, _ = load_model(args.precision)
        
        # Apply pruning
        pruned_transformer = apply_pruning(
            pruned_transformer, 
            method=config["method"],
            amount=config["amount"],
            structured=config["structured"]
        )
        
        # Get model size
        pruned_size = get_model_size(pruned_transformer)
        pruned_pipeline = create_pipeline(pruned_transformer)
        
        # Benchmark the pruned model
        pruned_dir = os.path.join(args.output_dir, config["name"])
        os.makedirs(pruned_dir, exist_ok=True)
        
        pruned_results, pruned_time = generate_and_benchmark(
            pruned_pipeline, test_prompts, args.steps, pruned_dir
        )
        
        # Store results
        results[config["name"]] = {
            "size": pruned_size,
            "time": pruned_time,
            "speedup": original_time / pruned_time if pruned_time > 0 else 0,
            "sparsity": get_sparsity(pruned_transformer),
            "size_reduction": (original_size - pruned_size) / original_size
        }
        
        print(f"Pruned model size: {pruned_size:.2f} MB")
        print(f"Size reduction: {100 * (original_size - pruned_size) / original_size:.1f}%")
        print(f"Average generation time: {pruned_time:.2f}s")
        print(f"Speedup: {results[config['name']]['speedup']:.2f}x")
    
    # Generate comparative visualization
    print("\n=== Generating Visualization ===")
    
    # Plot size comparison
    plt.figure(figsize=(12, 6))
    methods = ["original"] + [config["name"] for config in pruning_configs]
    sizes = [results[method]["size"] for method in methods]
    
    plt.subplot(1, 2, 1)
    plt.bar(methods, sizes)
    plt.title("Model Size Comparison")
    plt.ylabel("Size (MB)")
    plt.xticks(rotation=45)
    
    # Plot time comparison
    times = [results[method]["time"] for method in methods]
    plt.subplot(1, 2, 2)
    plt.bar(methods, times)
    plt.title("Generation Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "benchmark_comparison.png"))
    
    # Save detailed results
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Generate HTML report
    html_report = f"""
    <html>
    <head>
        <title>Diffusion Model Pruning Benchmark</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .comparison {{ display: flex; flex-wrap: wrap; }}
            .image-card {{ margin: 10px; text-align: center; }}
            img {{ max-width: 300px; }}
        </style>
    </head>
    <body>
        <h1>Diffusion Model Pruning Benchmark Results</h1>
        <h2>Configuration</h2>
        <ul>
            <li>Inference Steps: {args.steps}</li>
            <li>Precision: {precision}</li>
            <li>Number of Test Prompts: {len(test_prompts)}</li>
        </ul>
        
        <h2>Performance Comparison</h2>
        <table>
            <tr>
                <th>Method</th>
                <th>Size (MB)</th>
                <th>Size Reduction</th>
                <th>Sparsity</th>
                <th>Avg. Time (s)</th>
                <th>Speedup</th>
            </tr>
    """
    
    for method in methods:
        if method == "original":
            size_reduction = "0%"
        else:
            size_reduction = f"{100 * results[method]['size_reduction']:.1f}%"
        
        html_report += f"""
            <tr>
                <td>{method}</td>
                <td>{results[method]['size']:.2f}</td>
                <td>{size_reduction}</td>
                <td>{results[method]['sparsity']:.2f}%</td>
                <td>{results[method]['time']:.2f}</td>
                <td>{results[method]['speedup']:.2f}x</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>Visual Comparison</h2>
        <p>Comparison of generated images across different pruning methods:</p>
    """
    
    # Add image comparisons
    for i, prompt in enumerate(test_prompts):
        html_report += f"""
        <h3>Prompt {i+1}: "{prompt}"</h3>
        <div class="comparison">
        """
        
        for method in methods:
            img_path = f"{method}/image_{i}.png"
            html_report += f"""
            <div class="image-card">
                <img src="{img_path}" alt="{method} image">
                <p>{method}</p>
            </div>
            """
        
        html_report += "</div>"
    
    html_report += """
        <h2>Summary Chart</h2>
        <img src="benchmark_comparison.png" alt="Performance Comparison Chart" style="max-width: 100%">
    </body>
    </html>
    """
    
    with open(os.path.join(args.output_dir, "benchmark_report.html"), "w", encoding="utf-8") as f:
        f.write(html_report)
    
    print(f"\nBenchmark complete! Results saved to {args.output_dir}")
    print(f"See detailed report at {os.path.join(args.output_dir, 'benchmark_report.html')}")

if __name__ == "__main__":
    main()
