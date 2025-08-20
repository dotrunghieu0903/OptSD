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
import gc
import os
import sys

# Add parent directory to path for importing metrics and resizing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import calculate_fid, compute_image_reward, calculate_clip_score, calculate_lpips, calculate_psnr_resized
from resizing_image import resize_images

# Configure PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    # Create the pipeline with memory optimizations
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to("cuda")
    
    # Enable memory efficient attention if available
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing(1)
    
    return pipeline

def generate_image(pipeline, prompt, num_inference_steps=50, guidance_scale=3.5, output_path=None):
    """
    Generate an image using the diffusion pipeline.
    
    Args:
        pipeline: The diffusion pipeline
        prompt: The text prompt for generation
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        output_path: Optional path to save the generated image
        
    Returns:
        Tuple of (generated image, generation time)
    """
    # Free up CUDA cache before generation
    torch.cuda.empty_cache()
    
    # Set PyTorch memory allocation config to allow memory growth
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of available memory
    
    start_time = time.time()
    with torch.cuda.amp.autocast(enabled=True):  # Enable automatic mixed precision
        # Check if height/width parameters are supported
        pipeline_params = {}
        import inspect
        sig = inspect.signature(pipeline.__call__)
        if 'height' in sig.parameters and 'width' in sig.parameters:
            pipeline_params.update({
                'height': 384,  # Smaller dimensions to reduce memory usage
                'width': 384
            })
        
        image = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **pipeline_params
        ).images[0]
    generation_time = time.time() - start_time
    
    # Free memory again after generation
    torch.cuda.empty_cache()
    
    # Save the image if an output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"\n✅ Image saved to: {output_path}")
        print(f"========================")
        print(f"IMAGE SAVED TO: {output_path}")
        print(f"========================")
    
    return image, generation_time

def setup_memory_optimizations():
    """
    Configure memory optimizations for PyTorch to avoid CUDA OOM errors.
    """
    # Clear any cached memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Print available GPU memory for debugging
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory = gpu_properties.total_memory / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)
        print(f"GPU: {gpu_properties.name}")
        print(f"Total GPU memory: {total_memory:.2f} GiB")
        print(f"Allocated GPU memory: {allocated_memory:.2f} GiB")
        print(f"Reserved GPU memory: {reserved_memory:.2f} GiB")
        print(f"Free GPU memory: {total_memory - allocated_memory:.2f} GiB")
    
    # Set memory fraction to avoid OOM
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.9)

def main():
    # Setup memory optimizations to avoid CUDA OOM errors
    setup_memory_optimizations()
    
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
    parser.add_argument("--image_size", type=int, default=512,
                        help="Size of the generated image (smaller = less memory)")
    parser.add_argument("--coco_dir", type=str, default="coco",
                        help="Path to COCO dataset (for FID calculation)")
    parser.add_argument("--calculate_metrics", action="store_true",
                        help="Whether to calculate image quality metrics")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = "combined_quant_pruning_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the model with original weights
    print("\n=== Loading Original Model ===")
    transformer, precision = load_model(args.precision)
    original_size = get_model_size(transformer)
    print(f"Original model size: {original_size:.2f} MB")
    
    # Create pipeline and generate baseline image
    pipeline = create_pipeline(transformer)
    print("\n=== Generating Baseline Image ===")
    # Clear memory before generation
    torch.cuda.empty_cache()
    gc.collect()
    
    baseline_image, baseline_time = generate_image(
        pipeline, args.prompt, args.steps, 
        output_path=os.path.join(output_dir, "baseline_image.png")
    )
    print(f"Baseline generation time: {baseline_time:.2f}s")
    
    # Force release memory
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    
    # 2. Apply pruning to the model
    print("\n=== Applying Pruning ===")
    pruned_transformer = apply_magnitude_pruning(transformer, amount=args.pruning_amount)
    pruned_size = get_model_size(pruned_transformer)
    print(f"Pruned model size: {pruned_size:.2f} MB")
    print(f"Size reduction: {original_size - pruned_size:.2f} MB ({100 * (original_size - pruned_size) / original_size:.1f}%)")
    
    # Create pruned pipeline and generate pruned image
    # Clear memory before creating the pruned pipeline
    torch.cuda.empty_cache()
    gc.collect()
    
    pruned_pipeline = create_pipeline(pruned_transformer)
    print("\n=== Generating Pruned Image ===")
    pruned_image, pruned_time = generate_image(
        pruned_pipeline, args.prompt, args.steps,
        output_path=os.path.join(output_dir, f"pruned_{int(args.pruning_amount*100)}pct_image.png")
    )
    print(f"Pruned generation time: {pruned_time:.2f}s")
    print(f"Speed improvement: {(baseline_time - pruned_time) / baseline_time * 100:.1f}%")
    
    # Release memory
    del pruned_pipeline
    torch.cuda.empty_cache()
    gc.collect()
    
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
    
    # 4. Calculate and report metrics if requested
    metrics_results = {}
    
    if args.calculate_metrics:
        print("\n=== Calculating Image Quality Metrics ===")
        
        # Create a directory for resized images
        resized_output_dir = os.path.join(output_dir, "resized")
        os.makedirs(resized_output_dir, exist_ok=True)
        
        # Save original and pruned images for metrics calculation
        baseline_filename = "baseline_image.png"
        pruned_filename = f"pruned_{int(args.pruning_amount*100)}pct_image.png"
        
        # Create a dictionary with original dimensions
        # For this example, we're using the same dimensions for both images
        image_dimensions = {
            baseline_filename: (512, 512),
            pruned_filename: (512, 512)
        }
        
        # Create captions dictionary for CLIP Score and ImageReward
        captions = {
            baseline_filename: args.prompt,
            pruned_filename: args.prompt
        }
        
        # Resize images for metrics calculation
        resize_images(output_dir, resized_output_dir, image_dimensions)
        
        # Calculate FID score if COCO dataset is available
        coco_val_dir = os.path.join(args.coco_dir, "val2017")
        if os.path.exists(coco_val_dir):
            try:
                print("\n--- Calculating FID Score ---")
                calculate_fid(output_dir, resized_output_dir, coco_val_dir)
            except Exception as e:
                print(f"Error calculating FID: {e}")
        
        # Calculate CLIP Score
        try:
            clip_score = calculate_clip_score(output_dir, captions)
            metrics_results["clip_score"] = clip_score
        except Exception as e:
            print(f"Error calculating CLIP Score: {e}")
        
        # Calculate ImageReward
        try:
            image_reward = compute_image_reward(output_dir, captions)
            metrics_results["image_reward"] = image_reward
        except Exception as e:
            print(f"Error calculating ImageReward: {e}")
        
        # Calculate LPIPS between baseline and pruned images
        try:
            print("\n--- Calculating LPIPS ---")
            # Create a list of filenames for LPIPS calculation
            filenames = [pruned_filename]
            # Using resized images for LPIPS calculation to ensure same dimensions
            lpips_score = calculate_lpips(resized_output_dir, resized_output_dir, filenames)
            metrics_results["lpips"] = lpips_score
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")
        
        # Calculate PSNR between baseline and pruned images
        try:
            print("\n--- Calculating PSNR ---")
            # We need both original and generated images to have the same dimensions for PSNR
            psnr_score = calculate_psnr_resized(resized_output_dir, resized_output_dir, [pruned_filename])
            metrics_results["psnr"] = psnr_score
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
    
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
        f.write(f"Speed improvement: {(baseline_time - pruned_time) / baseline_time * 100:.1f}%\n\n")
        
        # Add metrics to summary if calculated
        if metrics_results:
            f.write("=== Image Quality Metrics ===\n")
            for metric_name, metric_value in metrics_results.items():
                if metric_value is not None:
                    f.write(f"{metric_name.upper()}: {metric_value:.4f}\n")
    
    print(f"Results saved to {output_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
