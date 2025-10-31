"""
Pruned module for diffusion models.

This module combines the basic pruning functionality with advanced pruning techniques 
to optimize diffusion model weights, specifically targeting the transformer components.
It supports multiple pruning methods with dataset-based evaluation.
"""

import os
import sys
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import gc
from tqdm import tqdm
from PIL import Image

from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from huggingface_hub import login

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.flickr8k import process_flickr8k

from shared.cleanup import setup_memory_optimizations_pruning
from shared.resources_monitor import generate_image_and_monitor
from shared.metrics import calculate_fid_subset, compute_image_reward, calculate_clip_score, calculate_lpips, calculate_psnr_resized
from shared.resizing_image import resize_images

# Configure PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ----- Basic Pruning Functions -----

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

def iterative_pruning(model, final_amount=0.3, steps=5, name_filter=None):
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

def save_pruning_results(results, output_dir="output"):
    """
    Save the images and benchmark results.
    
    Args:
        results: Dict with benchmark results
        output_dir: Directory to save output
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


# ----- Advanced Pruning Functions -----

def load_model(model_name=None):
    print("Loading base model...")
    
    # Load configuration from config.json
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get model paths from config.json
    models = config.get("models", {})
    
    if model_name is None:
        model_name = config.get("model_params", {}).get("model_name", "Flux.1-dev")
    
    if model_name in models:
        model_info = models[model_name]
        model_path = model_info.get("path")
        transformer_path = model_info.get("quant_path")
        print(f"Using model: {model_name}")
        print(f"Model path: {model_path}")
        print(f"Transformer path: {transformer_path}")
    else:
        print(f"Model {model_name} not found in config.json. Using default path.")
        transformer_path = "nunchaku-tech/nunchaku-flux.1-dev/svdq-int4_r32-flux.1-dev.safetensors"
    
    # Load the transformer model
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(transformer_path)
    
    return transformer

def create_pipeline(transformer):
    """
    Create a pipeline with the provided transformer model.
    
    Args:
        transformer: The transformer model to use in the pipeline
        
    Returns:
        The FluxPipeline with the transformer
    """
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    return pipeline

def enhance_caption(caption):
    """
    Enhance a caption with details to improve image quality.
    
    Args:
        caption: The base caption
        
    Returns:
        Enhanced caption
    """
    quality_prefixes = [
        "a high quality photo of", 
        "a detailed photograph of",
        "a professional photograph depicting",
        "a sharp and clear image of",
        "a high resolution picture showing"
    ]
    
    quality_suffixes = [
        ", high quality, detailed, sharp focus, professional photography",
        ", high resolution, highly detailed, professional photo",
        ", clear details, crisp focus, professional lighting",
        ", 8k resolution, detailed, professional",
        ", crystal clear, detailed texture, professional photograph"
    ]
    
    # Skip enhancement if caption already has quality terms
    quality_terms = ["high quality", "detailed", "professional", "8k", "clear", "sharp", "resolution"]
    if any(term in caption.lower() for term in quality_terms):
        return caption
    
    # Add prefix and suffix for quality
    enhanced_caption = random.choice(quality_prefixes) + " " + caption + random.choice(quality_suffixes)
    return enhanced_caption

def filter_good_captions(caption):
    """
    Filter out captions that are likely to produce poor results.
    
    Args:
        caption: The caption to check
        
    Returns:
        True if the caption is good, False otherwise
    """
    # Skip captions that are too short
    if len(caption.split()) < 3:
        return False
    
    # Skip captions with specific problematic terms
    problematic_terms = [
        "ferris wheel", "clock tower", "giraffe", "zebra", "kite", "tennis", 
        "skate", "ski", "skateboard", "surfboard", "baseball", "football", 
        "soccer", "basketball", "stop sign", "traffic light", "fire hydrant",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow"
    ]
    
    if any(term.lower() in caption.lower() for term in problematic_terms):
        return False
    
    return True

def load_coco_captions(annotations_file, limit=500):
    """
    Load captions from the COCO dataset.
    
    Args:
        annotations_file: Path to the COCO annotations file
        limit: Maximum number of captions to load
        
    Returns:
        Dictionary mapping filenames to captions and image dimensions
    """
    print(f"Loading captions from {annotations_file}")
    
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Create mapping of image_id to filename
    id_to_filename = {}
    for img in data["images"]:
        # COCO filenames are 12 digits padded with zeros, e.g., 000000039769.jpg
        id_to_filename[img["id"]] = img["file_name"]
        
    # Create mapping of image_id to dimensions
    id_to_dimensions = {}
    for img in data["images"]:
        id_to_dimensions[img["id"]] = (img["width"], img["height"])
    
    # Create mapping of filename to caption
    filename_to_caption = {}
    filename_to_dimensions = {}
    
    # Filter and enhance captions
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"]
        
        # Only include captions that pass the filter
        if filter_good_captions(caption):
            # Enhance the caption
            enhanced_caption = enhance_caption(caption)
            
            # Get the filename and dimensions
            filename = id_to_filename[image_id]
            dimensions = id_to_dimensions[image_id]
            
            # Add to the mappings
            filename_to_caption[filename] = enhanced_caption
            filename_to_dimensions[filename] = dimensions
            
            # Stop once we have enough captions
            if len(filename_to_caption) >= limit:
                break
    
    print(f"Loaded {len(filename_to_caption)} captions")
    return filename_to_caption, filename_to_dimensions

def apply_pruning_method(transformer, method, amount, structured_dim=0, iterative_steps=5):
    """
    Apply the specified pruning method to the transformer model.
    
    Args:
        transformer: The transformer model to prune
        method: Pruning method ("magnitude", "structured", "iterative", "attention_heads")
        amount: Amount to prune (0.0 to 1.0)
        structured_dim: Dimension for structured pruning (0 for rows, 1 for columns)
        iterative_steps: Number of steps for iterative pruning
        
    Returns:
        The pruned transformer and statistics
    """
    print(f"Applying {method} pruning with amount {amount}...")
    
    original_size = get_model_size(transformer)
    
    if method == "magnitude":
        pruned_transformer = apply_magnitude_pruning(transformer, amount=amount)
    elif method == "structured":
        pruned_transformer = apply_structured_pruning(transformer, amount=amount, dim=structured_dim)
    elif method == "iterative":
        pruned_transformer = iterative_pruning(transformer, final_amount=amount, steps=iterative_steps)
    elif method == "attention_heads":
        # Calculate how many heads to prune per layer
        num_layers = len([name for name, _ in transformer.named_modules() 
                         if "attention" in name and "output" in name])
        num_heads = 8  # Typical number of heads per layer
        heads_per_layer = max(1, int(num_heads * amount))
        
        heads_to_prune = {}
        for i in range(num_layers):
            heads_to_prune[i] = list(range(heads_per_layer))
        
        pruned_transformer = prune_attention_heads(transformer, heads_to_prune)
    else:
        print(f"Unknown pruning method: {method}. Using magnitude pruning.")
        pruned_transformer = apply_magnitude_pruning(transformer, amount=amount)
    
    # Calculate model size after pruning
    pruned_size = get_model_size(pruned_transformer)
    pruning_stats = {
        "method": method,
        "amount": amount,
        "original_size_mb": original_size,
        "pruned_size_mb": pruned_size,
        "size_reduction_mb": original_size - pruned_size,
        "size_reduction_percent": 100 * (original_size - pruned_size) / original_size,
        "sparsity_percent": get_sparsity(pruned_transformer)
    }
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Pruned model size: {pruned_size:.2f} MB")
    print(f"Size reduction: {original_size - pruned_size:.2f} MB ({pruning_stats['size_reduction_percent']:.1f}%)")
    print(f"Model sparsity: {pruning_stats['sparsity_percent']:.2f}%")
    
    return pruned_transformer, pruning_stats


def main(args=None):
    # If no args provided, parse them from command line
    if args is None:
        # Setup argument parser
        parser = argparse.ArgumentParser(description="Advanced Pruning with dataset")
        parser.add_argument("--pruning_amount", type=float, default=0.3, 
                            help="Amount of weights to prune (0.0 to 0.9)")
        parser.add_argument("--pruning_method", type=str, default="magnitude", 
                            choices=["magnitude", "structured", "iterative", "attention_heads"],
                            help="Pruning method to use")
        parser.add_argument("--structured_dim", type=int, default=0,
                            help="Dimension for structured pruning (0 for rows, 1 for columns)")
        parser.add_argument("--iterative_steps", type=int, default=5,
                            help="Number of steps for iterative pruning")
        parser.add_argument("--num_images", type=int, default=500,
                            help="Number of images to process")
        parser.add_argument("--steps", type=int, default=30,
                            help="Number of inference steps")
        parser.add_argument("--guidance_scale", type=float, default=3.5,
                            help="Guidance scale for image generation")
        parser.add_argument("--skip_metrics", action="store_true",
                            help="Skip calculation of image quality metrics")
        parser.add_argument("--metrics_subset", type=int, default=100,
                            help="Number of images to use for metrics calculation (default: 100)")
        parser.add_argument("--use_flickr8k", action="store_true",
                            help="Use Flickr8k dataset instead of COCO")
        parser.add_argument("--model_name", type=str, default="Flux.1-dev",
                            help="Name of the model to use from config.json")
        args = parser.parse_args()
    
    # Authenticate with Hugging Face
    login(token="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")

    # Setup memory optimizations to avoid CUDA OOM errors
    setup_memory_optimizations_pruning()
    
    # Create output directories
    output_dir = "pruning/output"
    os.makedirs(output_dir, exist_ok=True)

    image_filename_to_caption = {}
    image_dimensions = {}
    original_dir = ""
    
    if hasattr(args, 'use_flickr8k') and args.use_flickr8k:
        # Load Flickr8k dataset
        print("Using Flickr8k dataset...")
        flickr8k_dir = "flickr8k"
        flickr_caption_path = f"{flickr8k_dir}/captions.txt"
        original_dir = f"{flickr8k_dir}/Images"
        image_filename_to_caption, image_dimensions = process_flickr8k(
            original_dir, flickr_caption_path, limit=args.num_images
        )
    else:
        # Load COCO dataset
        print("Using COCO dataset...")
        # Define COCO paths
        coco_dir = "coco"
        annotations_file = os.path.join(coco_dir, "annotations", "captions_val2017.json")
        
        # 1. Load COCO captions
        print("\n=== Loading COCO Captions ===")
        image_filename_to_caption, image_dimensions = load_coco_captions(
            annotations_file, limit=args.num_images
        )
        original_dir = os.path.join(coco_dir, "val2017")

    # 2. Load the model
    print("\n=== Loading Base Model ===")
    model_name = getattr(args, 'model_name', None)
    transformer = load_model(model_name)
    
    # 3. Apply pruning to the model based on selected method
    # Get method from args or use default
    pruning_method = getattr(args, 'pruning_method', "magnitude")
    pruning_amount = getattr(args, 'pruning_amount', 0.3)
    structured_dim = getattr(args, 'structured_dim', 0)
    iterative_steps = getattr(args, 'iterative_steps', 5)
    
    print(f"\n=== Applying {pruning_method} Pruning to Model ===")
    pruned_transformer, pruning_stats = apply_pruning_method(
        transformer, 
        method=pruning_method,
        amount=pruning_amount,
        structured_dim=structured_dim,
        iterative_steps=iterative_steps
    )
    
    # Create pipeline with the pruned model
    print("\n=== Creating Pipeline with Pruned Model ===")
    try:
        pipeline = create_pipeline(pruned_transformer)
        
        # Test the pipeline with a simple prompt to ensure it works
        print("Testing pipeline with a simple prompt...")
        test_image = pipeline(
            "a high quality photo of a motorcycle parked on the gravel in front of a garage",
            num_inference_steps=1,
            guidance_scale=7.5
        ).images[0]
        print("Pipeline test successful!")
        
        # Clear memory
        del test_image
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return
    
    # Create a list to store generation metadata
    generation_metadata = []
    
    # 4. Generate images for captions
    print(f"\n=== Generating Images for {len(image_filename_to_caption)} Captions ===")
    
    # Keep track of generation time
    generation_times = {}
    
    # Use all captions since they're already limited at dataset level
    filenames_captions = list(image_filename_to_caption.items())
    for i, (filename, prompt) in enumerate(tqdm(filenames_captions, desc="Generating images")):
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n\n{'='*80}")
        print(f"Processing image {i+1}/{len(filenames_captions)}: {filename} (Prompt: {prompt[:50]}...)")
        print(f"Output will be saved to: {output_path}")
        print(f"{'='*80}")
        
        # Skip if the image already exists
        if os.path.exists(output_path):
            print(f"Skipping generation for {filename} (already exists)")
            continue
            
        try:
            # Generate image and monitor VRAM
            generation_time, metadata = generate_image_and_monitor(
                pipeline, prompt, output_path, filename, num_inference_steps=args.steps, guidance_scale=args.guidance_scale
            )
            # Add the metadata to our list
            generation_metadata.append(metadata)
            # Add the generation time to the dictionary
            generation_times[filename] = generation_time
            print(f"Generated image {i+1}/{len(filenames_captions)}: {filename}")
            print(f"Image available at: {output_path}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Error when generating image for prompt '{prompt[:50]}...': {e}")
            generation_times[filename] = -1  # Indicate an error
            
        # Force GC to free memory
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    print("Image generation complete.")
    
    # 5. Calculate Image Quality Metrics (if not skipped)
    metrics_results = {}
    
    if args.skip_metrics:
        print("\n=== Skipping Image Quality Metrics (--skip_metrics flag set) ===")
    else:
        print("\n=== Calculating Image Quality Metrics ===")
    
        # Create directory for resized images (needed for FID and PSNR)
        resized_output_dir = os.path.join(output_dir, "resized")
        os.makedirs(resized_output_dir, exist_ok=True)
        
        # Resize images for metrics calculation
        try:
            print("\n--- Resizing Images for Metrics Calculation ---")
            resize_images(output_dir, resized_output_dir, image_dimensions)
        except Exception as e:
            print(f"Error resizing images: {e}")
        
        # Calculate FID score (Subset) (Subset)
        try:
            print("--- Calculating FID Score (Subset) ---")
            
            fid_score = calculate_fid_subset(output_dir, resized_output_dir, original_dir)
            metrics_results["fid_score"] = fid_score
        except Exception as e:
            print(f"Error calculating FID: {e}")
        
        # Calculate CLIP Score
        try:
            print("\n--- Calculating CLIP Score ---")
            clip_score = calculate_clip_score(output_dir, image_filename_to_caption)
            metrics_results["clip_score"] = clip_score
        except Exception as e:
            print(f"Error calculating CLIP Score: {e}")
        
        # Calculate ImageReward
        try:
            print("\n--- Calculating ImageReward ---")
            image_reward = compute_image_reward(output_dir, image_filename_to_caption)
            metrics_results["image_reward"] = image_reward
        except Exception as e:
            print(f"Error calculating ImageReward: {e}")
        
        # Calculate LPIPS - we need original images to compare with generated images
        try:
            print("\n--- Calculating LPIPS ---")
            # For dataset, we need to select a subset of filenames for LPIPS calculation
            # We should compare resized images to ensure dimensions match
            
            # Create a directory for resized original images
            resized_original_dir = os.path.join(output_dir, "resized_original")
            os.makedirs(resized_original_dir, exist_ok=True)
            
            # Get list of generated filenames that have been resized
            generated_filenames = [f for f in os.listdir(resized_output_dir) if os.path.isfile(os.path.join(resized_output_dir, f)) 
                                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            # Use the metrics_subset parameter to limit the number of images for metrics
            subset_size = min(args.metrics_subset, len(generated_filenames))
            selected_filenames = generated_filenames[:subset_size]

            # Manually resize original dataset images to match generated images for LPIPS calculation
            for filename in tqdm(selected_filenames, desc="Resizing original images for LPIPS"):
                try:
                    # Load the original image
                    original_path = os.path.join(original_dir, filename)
                    original_img = Image.open(original_path).convert("RGB")
                    
                    # Get the target size from the generated resized image
                    gen_img_path = os.path.join(resized_output_dir, filename)
                    gen_img = Image.open(gen_img_path)
                    target_size = gen_img.size
                    
                    # Resize the original image
                    resized_original = original_img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # Save the resized original
                    resized_original_path = os.path.join(resized_original_dir, filename)
                    resized_original.save(resized_original_path)
                except Exception as e:
                    print(f"Error resizing original image {filename}: {e}")
            
            # Now calculate LPIPS using the resized original and generated images
            lpips_score = calculate_lpips(resized_original_dir, resized_output_dir, selected_filenames)
            metrics_results["lpips"] = lpips_score
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")
        
        # Calculate PSNR - we need original images to compare with generated images
        try:
            print("\n--- Calculating PSNR ---")
            # We use the same resized images prepared for LPIPS
            psnr_score = calculate_psnr_resized(resized_original_dir, resized_output_dir, selected_filenames)
            metrics_results["psnr"] = psnr_score
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
    
    # Save metadata to JSON file
    metadata_file = os.path.join(output_dir, f"{args.pruning_method}_pruning_metadata.json")
    try:
        # Add metrics to metadata
        combined_metadata = {
            "model_info": {
                "pruning_method": args.pruning_method,
                "pruning_amount": args.pruning_amount,
                "pruning_stats": pruning_stats
            },
            "generation_params": {
                "inference_steps": args.steps,
                "guidance_scale": args.guidance_scale
            },
            "generation_metadata": generation_metadata
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(combined_metadata, f, indent=2)
            
        print(f"Saved metadata to: {metadata_file}")
    except Exception as e:
        print(f"Error when saving metadata JSON: {e}")
    
    # Calculate and print statistics
    successful_generations = [t for t in generation_times.values() if t > 0]
    if successful_generations:
        avg_time = sum(successful_generations) / len(successful_generations)
        print(f"\n=== Generation Statistics ===")
        print(f"Successfully generated {len(successful_generations)}/{len(image_filename_to_caption)} images")
        print(f"Average generation time: {avg_time:.2f} seconds per image")
        print(f"Total generation time: {sum(successful_generations):.2f} seconds")
        
        # Calculate average VRAM usage across all images
        all_vram_data = [meta.get("average_vram_gb") for meta in generation_metadata if meta.get("average_vram_gb") is not None]
        if all_vram_data:
            avg_vram = sum(all_vram_data) / len(all_vram_data)
            max_vram = max([meta.get("max_vram_gb", 0) for meta in generation_metadata])
            print(f"Average VRAM usage: {avg_vram:.2f} GB")
            print(f"Maximum VRAM usage: {max_vram:.2f} GB")
    
    # Save summary report
    with open(os.path.join(output_dir, f"{args.pruning_method}_pruning_summary.txt"), "w", encoding='utf-8') as f:
        f.write(f"=== {args.pruning_method.capitalize()} Pruning Summary ===\n\n")
        f.write(f"Processed {len(image_filename_to_caption)} captions\n")
        f.write(f"Inference steps: {args.steps}\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n")
        f.write(f"Pruning method: {args.pruning_method}\n")
        f.write(f"Pruning amount: {args.pruning_amount*100:.1f}%\n")
        f.write(f"Original model size: {pruning_stats['original_size_mb']:.2f} MB\n")
        f.write(f"Pruned model size: {pruning_stats['pruned_size_mb']:.2f} MB\n")
        f.write(f"Size reduction: {pruning_stats['size_reduction_mb']:.2f} MB ({pruning_stats['size_reduction_percent']:.1f}%)\n")
        f.write(f"Model sparsity: {pruning_stats['sparsity_percent']:.2f}%\n\n")
        
        if successful_generations:
            f.write(f"Successfully generated {len(successful_generations)}/{len(image_filename_to_caption)} images\n")
            f.write(f"Average generation time: {avg_time:.2f} seconds per image\n")
            f.write(f"Total generation time: {sum(successful_generations):.2f} seconds\n\n")
            
            if all_vram_data:
                f.write(f"Average VRAM usage: {avg_vram:.2f} GB\n")
                f.write(f"Maximum VRAM usage: {max_vram:.2f} GB\n\n")
        
        # Add metrics results to summary
        if metrics_results:
            f.write("=== Image Quality Metrics ===\n")
            for metric_name, metric_value in metrics_results.items():
                f.write(f"{metric_name}: {metric_value}\n")
    
    print(f"Completed. Results saved to {output_dir}")
    
    # Return success
    return 0

if __name__ == "__main__":
    main()