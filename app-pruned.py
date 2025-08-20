"""
Example application demonstrating model pruning for diffusion models.
This script applies pruning to a diffusion model and demonstrates the impact on 
performance and image quality.
"""

import torch
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision
import os
import time
from tqdm import tqdm
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.utils.prune as prune

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

def apply_magnitude_pruning(model, amount=0.3):
    """
    Apply magnitude pruning to the model parameters.
    
    Args:
        model: The model to prune
        amount: The fraction of parameters to prune (0.0 to 1.0)
        
    Returns:
        The pruned model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make pruning permanent by removing the mask
            prune.remove(module, 'weight')
    
    print(f"Applied {amount*100:.1f}% magnitude pruning to model")
    print(f"Model sparsity after pruning: {get_sparsity(model):.2f}%")
    return model

def load_model(precision=None):
    """
    Load the diffusion model with the specified precision.
    
    Args:
        precision: The precision to use (fp4, fp8, etc.). If None, will use auto-detected.
        
    Returns:
        The loaded transformer model and precision used
    """
    if precision is None:
        # Auto-detect precision
        precision = get_precision()
        print(f"Detected precision: {precision}")
    
    # Load the transformer model
    print(f"Loading model with {precision} precision...")
    model_path = f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(model_path)
    
    return transformer, precision

def create_pipeline(transformer):
    """
    Create a diffusion pipeline using the provided transformer.
    
    Args:
        transformer: The transformer model
        
    Returns:
        The diffusion pipeline
    """
    # Create the pipeline
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    return pipeline

def generate_image(pipeline, prompt, num_inference_steps=50, guidance_scale=3.5):
    """
    Generate an image using the diffusion pipeline.
    
    Args:
        pipeline: The diffusion pipeline
        prompt: The text prompt for generation
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        
    Returns:
        Tuple of (generated image, generation time)
    """
    start_time = time.time()
    image = pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]
    generation_time = time.time() - start_time
    
    return image, generation_time

def main():
    parser = argparse.ArgumentParser(description="Pruned diffusion model demo")
    parser.add_argument("--pruning_amount", type=float, default=0.3, 
                        help="Amount of weights to prune (0.0 to 0.9)")
    parser.add_argument("--precision", type=str, default=None, 
                        help="Precision to use (fp4, fp8, fp16, etc.)")
    parser.add_argument("--prompt", type=str, 
                        default="A futuristic city with tall skyscrapers",
                        help="Text prompt for image generation")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of inference steps")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = "pruned_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the model with original weights
    print("\n=== Loading Original Model ===")
    transformer, precision = load_model(args.precision)
    original_size = get_model_size(transformer)
    print(f"Original model size: {original_size:.2f} MB")
    
    # Create pipeline and generate baseline image
    pipeline = create_pipeline(transformer)
    print("\n=== Generating Baseline Image ===")
    baseline_image, baseline_time = generate_image(
        pipeline, args.prompt, args.steps
    )
    baseline_image.save(os.path.join(output_dir, "baseline_image.png"))
    print(f"Baseline generation time: {baseline_time:.2f}s")
    
    # 2. Apply pruning to the model
    print("\n=== Applying Pruning ===")
    pruned_transformer = apply_magnitude_pruning(transformer, amount=args.pruning_amount)
    pruned_size = get_model_size(pruned_transformer)
    print(f"Pruned model size: {pruned_size:.2f} MB")
    print(f"Size reduction: {original_size - pruned_size:.2f} MB ({100 * (original_size - pruned_size) / original_size:.1f}%)")
    
    # Create pruned pipeline and generate pruned image
    pruned_pipeline = create_pipeline(pruned_transformer)
    print("\n=== Generating Pruned Image ===")
    pruned_image, pruned_time = generate_image(
        pruned_pipeline, args.prompt, args.steps
    )
    pruned_image.save(os.path.join(output_dir, f"pruned_{int(args.pruning_amount*100)}pct_image.png"))
    print(f"Pruned generation time: {pruned_time:.2f}s")
    print(f"Speed improvement: {(baseline_time - pruned_time) / baseline_time * 100:.1f}%")
    
    # 3. Save comparison results
    print("\n=== Saving Results ===")
    # Create side-by-side comparison
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(baseline_image)
    axs[0].set_title(f"Original (Size: {original_size:.1f}MB, Time: {baseline_time:.2f}s)")
    axs[0].axis('off')
    
    axs[1].imshow(pruned_image)
    axs[1].set_title(f"Pruned {args.pruning_amount*100:.0f}% (Size: {pruned_size:.1f}MB, Time: {pruned_time:.2f}s)")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"))
    
    # Save text summary
    with open(os.path.join(output_dir, "pruning_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Diffusion Model Pruning Summary ===\n\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Inference steps: {args.steps}\n")
        f.write(f"Pruning amount: {args.pruning_amount*100:.1f}%\n\n")
        f.write("=== Original Model ===\n")
        f.write(f"Size: {original_size:.2f} MB\n")
        f.write(f"Generation time: {baseline_time:.2f}s\n\n")
        f.write("=== Pruned Model ===\n")
        f.write(f"Size: {pruned_size:.2f} MB\n")
        f.write(f"Generation time: {pruned_time:.2f}s\n")
        f.write(f"Size reduction: {original_size - pruned_size:.2f} MB ({100 * (original_size - pruned_size) / original_size:.1f}%)\n")
        f.write(f"Speed improvement: {(baseline_time - pruned_time) / baseline_time * 100:.1f}%\n")
    
    print(f"Results saved to {output_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
