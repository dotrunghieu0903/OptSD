"""
Pruning module for diffusion models.

This module provides functions to apply pruning to diffusion model weights,
specifically targeting the transformer components in models like FluxPipeline.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel
import time
import os
from tqdm import tqdm

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

def apply_magnitude_pruning(model, amount=0.3, name_filter=None):
    """
    Apply magnitude pruning to the model parameters.
    
    Args:
        model: The model to prune
        amount: The fraction of parameters to prune (0.0 to 1.0)
        name_filter: Optional string to filter parameter names
        
    Returns:
        The pruned model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            if name_filter is None or name_filter in name:
                prune.l1_unstructured(module, name='weight', amount=amount)
                # Make pruning permanent by removing the mask
                prune.remove(module, 'weight')
    
    print(f"Applied {amount*100:.1f}% magnitude pruning to model")
    print(f"Model sparsity after pruning: {get_sparsity(model):.2f}%")
    return model

def apply_structured_pruning(model, amount=0.3, dim=0, name_filter=None):
    """
    Apply structured pruning to the model parameters.
    
    Args:
        model: The model to prune
        amount: The fraction of parameters to prune (0.0 to 1.0)
        dim: The dimension to prune (0 for row pruning, 1 for column pruning)
        name_filter: Optional string to filter parameter names
        
    Returns:
        The pruned model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            if name_filter is None or name_filter in name:
                prune.ln_structured(module, name='weight', amount=amount, 
                                   n=2, dim=dim)  # L2 norm along dimension
                # Make pruning permanent by removing the mask
                prune.remove(module, 'weight')
    
    print(f"Applied {amount*100:.1f}% structured pruning to model (dim={dim})")
    print(f"Model sparsity after pruning: {get_sparsity(model):.2f}%")
    return model

def iterative_pruning(model, final_amount=0.5, steps=5, name_filter=None):
    """
    Apply iterative magnitude pruning to the model.
    
    Args:
        model: The model to prune
        final_amount: The final fraction to prune (0.0 to 1.0)
        steps: Number of pruning steps
        name_filter: Optional string to filter parameter names
        
    Returns:
        The pruned model
    """
    step_amount = 1 - (1 - final_amount) ** (1 / steps)
    
    original_size = get_model_size(model)
    print(f"Original model size: {original_size:.2f} MB")
    
    for i in range(steps):
        print(f"Pruning step {i+1}/{steps}")
        model = apply_magnitude_pruning(model, amount=step_amount, name_filter=name_filter)
        
    final_size = get_model_size(model)
    print(f"Final model size: {final_size:.2f} MB")
    print(f"Size reduction: {original_size - final_size:.2f} MB ({100 * (original_size - final_size) / original_size:.1f}%)")
    
    return model

def prune_attention_heads(model, heads_to_prune):
    """
    Prune specific attention heads in a transformer model.
    
    Args:
        model: The transformer model
        heads_to_prune: Dict mapping layer indices to list of heads to prune
        
    Returns:
        The pruned model
    """
    if hasattr(model, "prune_heads") and callable(model.prune_heads):
        model.prune_heads(heads_to_prune)
        print(f"Pruned attention heads: {heads_to_prune}")
    else:
        print("Model doesn't support attention head pruning")
    
    return model

def load_and_prune_model(model_path, precision="fp4", pruning_amount=0.3, 
                         structured=False, method="magnitude"):
    """
    Load a transformer model and apply pruning.
    
    Args:
        model_path: Path to the model
        precision: Precision to use (fp4, fp8, etc.)
        pruning_amount: Amount of weights to prune
        structured: Whether to use structured pruning
        method: Pruning method ("magnitude", "random", "attention_heads")
        
    Returns:
        The pruned transformer model
    """
    print(f"Loading model from {model_path} with precision {precision}")
    
    # Load the transformer model
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(model_path)
    
    # Record size before pruning
    size_before = get_model_size(transformer)
    print(f"Model size before pruning: {size_before:.2f} MB")
    
    # Apply pruning based on method
    if method == "magnitude":
        if structured:
            transformer = apply_structured_pruning(transformer, amount=pruning_amount)
        else:
            transformer = apply_magnitude_pruning(transformer, amount=pruning_amount)
    elif method == "iterative":
        transformer = iterative_pruning(transformer, final_amount=pruning_amount, steps=5)
    elif method == "attention_heads":
        # Example: prune 20% of heads in each layer
        num_layers = len([name for name, _ in transformer.named_modules() 
                         if "attention" in name and "output" in name])
        num_heads = 8  # Typically 8 or 12 heads per layer, adjust as needed
        heads_per_layer = max(1, int(num_heads * pruning_amount))
        
        heads_to_prune = {}
        for i in range(num_layers):
            heads_to_prune[i] = list(range(heads_per_layer))
            
        transformer = prune_attention_heads(transformer, heads_to_prune)
        
    # Record size after pruning
    size_after = get_model_size(transformer)
    print(f"Model size after pruning: {size_after:.2f} MB")
    print(f"Size reduction: {size_before - size_after:.2f} MB ({100 * (size_before - size_after) / size_before:.1f}%)")
    
    return transformer

def create_pruned_pipeline(model_path, precision="fp4", pruning_amount=0.3,
                           structured=False, method="magnitude"):
    """
    Create a pipeline with a pruned transformer.
    
    Args:
        model_path: Path to the model
        precision: Precision to use
        pruning_amount: Amount of weights to prune
        structured: Whether to use structured pruning
        method: Pruning method
        
    Returns:
        The FluxPipeline with a pruned transformer
    """
    # Load and prune the transformer
    transformer = load_and_prune_model(
        model_path=model_path,
        precision=precision,
        pruning_amount=pruning_amount,
        structured=structured,
        method=method
    )
    
    # Create the pipeline with the pruned transformer
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    return pipeline

def benchmark_pruned_model(pipeline, prompts, num_inference_steps=50, guidance_scale=3.5):
    """
    Benchmark a pruned model on image generation.
    
    Args:
        pipeline: The diffusion pipeline
        prompts: List of prompts to test
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        
    Returns:
        Dict with benchmark results
    """
    results = {
        "times": [],
        "images": []
    }
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        start_time = time.time()
        
        # Generate the image
        image = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        results["times"].append(generation_time)
        results["images"].append(image)
        
        print(f"Prompt {i+1}/{len(prompts)}: Generated in {generation_time:.2f}s")
    
    avg_time = sum(results["times"]) / len(results["times"])
    print(f"Average generation time: {avg_time:.2f}s")
    
    return results

def save_pruning_results(results, output_dir="pruning/pruned_outputs"):
    """
    Save the images and benchmark results.
    
    Args:
        results: Dict with benchmark results
        output_dir: Directory to save outputs
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the images
    for i, image in enumerate(results["images"]):
        image_path = os.path.join(output_dir, f"pruned_image_{i}.png")
        image.save(image_path)
    
    # Save the benchmark results
    benchmark_path = os.path.join(output_dir, "benchmark_results.txt")
    with open(benchmark_path, "w") as f:
        f.write("Pruned Model Benchmark Results\n")
        f.write("--------------------------\n")
        f.write(f"Number of images: {len(results['images'])}\n")
        f.write(f"Average generation time: {sum(results['times']) / len(results['times']):.2f}s\n")
        f.write("\nIndividual generation times:\n")
        for i, t in enumerate(results["times"]):
            f.write(f"Image {i+1}: {t:.2f}s\n")
    
    print(f"Results saved to {output_dir}")
