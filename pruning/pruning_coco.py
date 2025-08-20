"""
Pruning evaluation with COCO dataset.
This script applies pruning techniques to a diffusion model
and evaluates performance on 500 captions from the COCO dataset.
It also calculates image quality metrics: FID, CLIP Score, ImageReward, LPIPS, and PSNR.
"""
import os
import sys
import gc
import json
import argparse
from tqdm import tqdm
from PIL import Image

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.resources_monitor import generate_image_and_monitor
from metrics import calculate_fid, compute_image_reward, calculate_clip_score, calculate_lpips, calculate_psnr_resized
from resizing_image import resize_images

# Import from pruning module
from pruning import get_model_size, get_sparsity, apply_magnitude_pruning, load_and_prune_model

import torch
from huggingface_hub import login
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel

# Configure PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Authenticate with Hugging Face
login(token="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")

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

def load_model():
    """
    Load the standard (non-quantized) diffusion model.
    
    Returns:
        The loaded transformer model
    """
    # Load the standard transformer model
    print(f"Loading standard model...")
    
    # Load the sharded model files
    model_path = "nunchaku-tech/nunchaku-flux.1-dev/svdq-int4_r32-flux.1-dev.safetensors"  # Using dev model
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(model_path, offload=True)
    
    return transformer

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
        "black-forest-labs/FLUX.1-dev",  # Using dev model
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to("cuda")
    
    # Enable memory efficient attention if available
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing(1)
    
    if hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
        
    return pipeline

def enhance_caption(caption):
    """
    Enhance a caption to make it more suitable for image generation.
    
    Args:
        caption: The original caption
        
    Returns:
        Enhanced caption
    """
    # Remove any dots at the end
    caption = caption.rstrip('.')
    
    # Add prefixes that help with generation quality
    enhanced_prefixes = [
        "a high quality photo of",
        "a detailed image showing",
        "a professional photograph of",
        "a clear picture depicting"
    ]
    
    # Select a random prefix
    import random
    prefix = random.choice(enhanced_prefixes)
    
    # Combine prefix with caption and ensure first letter of caption is lowercase
    if caption and caption[0].isupper():
        caption = caption[0].lower() + caption[1:]
    
    enhanced_caption = f"{prefix} {caption}"
    
    # Add suffix to improve quality
    enhanced_suffixes = [
        ", high detail, professional lighting",
        ", sharp focus, high resolution",
        ", vibrant colors, professional photograph",
        ", detailed, realistic, high quality"
    ]
    
    suffix = random.choice(enhanced_suffixes)
    enhanced_caption = enhanced_caption + suffix
    
    return enhanced_caption

def filter_good_captions(caption):
    """
    Filter out captions that might lead to poor generation.
    
    Args:
        caption: The caption to check
        
    Returns:
        True if the caption is good, False otherwise
    """
    # Skip very short captions
    if len(caption.split()) < 4:
        return False
    
    # Skip captions with unwanted patterns
    unwanted_patterns = [
        "this is", "there is", "these are", "this picture",
        "photo of", "picture of", "image of", "snapshot",
        "it is", "it's a", "its a"
    ]
    
    lower_caption = caption.lower()
    for pattern in unwanted_patterns:
        if pattern in lower_caption:
            return False
    
    return True

def load_coco_captions(annotations_file, limit=500):
    """
    Load captions from COCO dataset with filtering and enhancement.
    
    Args:
        annotations_file: Path to COCO annotations file
        limit: Maximum number of captions to load
        
    Returns:
        Dictionary mapping image filenames to captions
    """
    print(f"Loading COCO captions from {annotations_file}")
    
    with open(annotations_file, 'r') as f:
        captions_data = json.load(f)
    
    # Build mapping from image_id to dimensions and filename
    image_id_to_dimensions = {
        img['id']: (img['width'], img['height'], img['file_name'])
        for img in captions_data['images']
    }
    
    print(f"Loaded {len(captions_data['annotations'])} annotations from COCO")
    
    # Group captions by image_id
    captions_by_image = {}
    for annotation in captions_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        if image_id not in captions_by_image:
            captions_by_image[image_id] = []
        captions_by_image[image_id].append(caption)
    
    print(f"Found captions for {len(captions_by_image)} unique images")
    
    # Select the best caption for each image
    image_filename_to_caption = {}
    image_dimensions = {}  # Store {filename: (width, height)}
    processed_count = 0
    
    for image_id, captions in captions_by_image.items():
        if image_id not in image_id_to_dimensions:
            continue
            
        # Find the best caption (longest caption after filtering)
        good_captions = [c for c in captions if filter_good_captions(c)]
        if not good_captions:
            # If no good captions, use the longest one
            good_captions = captions
            
        # Sort by length and choose the longest
        best_caption = sorted(good_captions, key=len, reverse=True)[0]
        
        # Enhance the caption
        enhanced_caption = enhance_caption(best_caption)
        
        # Store the enhanced caption and dimensions
        width, height, original_filename = image_id_to_dimensions[image_id]
        image_filename_to_caption[original_filename] = enhanced_caption
        image_dimensions[original_filename] = (width, height)
        
        processed_count += 1
        if processed_count >= limit:
            break
    
    print(f"Selected and enhanced {len(image_filename_to_caption)} captions (limit: {limit})")
    
    # Print some examples of enhanced captions
    print("\nExample enhanced captions:")
    import itertools
    for filename, caption in itertools.islice(image_filename_to_caption.items(), 3):
        print(f"  - {filename}: {caption}")
    
    return image_filename_to_caption, image_dimensions

def main(args=None):
    # If no args provided, parse them from command line
    if args is None:
        # Setup argument parser
        parser = argparse.ArgumentParser(description="Pruning evaluation with COCO dataset")
        parser.add_argument("--pruning_amount", type=float, default=0.3, 
                            help="Amount of weights to prune (0.0 to 0.9)")
        parser.add_argument("--pruning_method", type=str, default="magnitude", 
                            choices=["magnitude", "structured", "iterative", "attention_heads"],
                            help="Pruning method to use")
        parser.add_argument("--num_images", type=int, default=500,
                            help="Number of COCO images to process")
        parser.add_argument("--steps", type=int, default=30,
                            help="Number of inference steps")
        parser.add_argument("--guidance_scale", type=float, default=3.5,
                            help="Guidance scale for image generation")
        parser.add_argument("--skip_metrics", action="store_true",
                            help="Skip calculation of image quality metrics")
        parser.add_argument("--metrics_subset", type=int, default=100,
                            help="Number of images to use for metrics calculation (default: 100)")
        args = parser.parse_args()
    
    # Setup memory optimizations to avoid CUDA OOM errors
    setup_memory_optimizations()
    
    # Create output directories
    output_dir = "pruning/pruned_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define COCO paths
    coco_dir = "coco"
    annotations_file = os.path.join(coco_dir, "annotations", "captions_val2017.json")
    
    # 1. Load COCO captions
    print("\n=== Loading COCO Captions ===")
    image_filename_to_caption, image_dimensions = load_coco_captions(
        annotations_file, limit=args.num_images
    )
    
    # 2. Load the standard model
    print("\n=== Loading Model ===")
    try:
        transformer = load_model()
        standard_size = get_model_size(transformer)
        print(f"Standard model size: {standard_size:.2f} MB")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 3. Apply pruning to the model
    print("\n=== Applying Pruning to Model ===")
    try:
        # Use apply_magnitude_pruning from the pruning module
        if args.pruning_method == "magnitude":
            pruned_transformer = apply_magnitude_pruning(transformer, amount=args.pruning_amount)
        elif args.pruning_method == "structured":
            from pruning import apply_structured_pruning
            pruned_transformer = apply_structured_pruning(transformer, amount=args.pruning_amount)
        elif args.pruning_method == "iterative":
            from pruning import iterative_pruning
            pruned_transformer = iterative_pruning(transformer, final_amount=args.pruning_amount, steps=5)
        elif args.pruning_method == "attention_heads":
            from pruning import prune_attention_heads
            num_layers = len([name for name, _ in transformer.named_modules() 
                            if "attention" in name and "output" in name])
            num_heads = 8  # Typically 8 or 12 heads per layer, adjust as needed
            heads_per_layer = max(1, int(num_heads * args.pruning_amount))
            
            heads_to_prune = {}
            for i in range(num_layers):
                heads_to_prune[i] = list(range(heads_per_layer))
                
            pruned_transformer = prune_attention_heads(transformer, heads_to_prune)
        
        pruned_size = get_model_size(pruned_transformer)
        sparsity = get_sparsity(pruned_transformer)
        print(f"Pruned model size: {pruned_size:.2f} MB")
        print(f"Model sparsity: {sparsity:.2f}%")
        print(f"Size reduction: {standard_size - pruned_size:.2f} MB ({100 * (standard_size - pruned_size) / standard_size:.1f}%)")
    except Exception as e:
        print(f"Error during pruning: {e}")
        return
    
    # Create pipeline with the pruned model
    print("\n=== Creating Pipeline ===")
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
        print("Cannot proceed without a working pipeline")
        return
    
    # Create a list to store generation metadata
    generation_metadata = []
    
    # 4. Generate images for COCO captions
    print(f"\n=== Generating Images for {len(image_filename_to_caption)} COCO Captions ===")
    
    # Keep track of generation time
    generation_times = {}
    
    for i, (filename, prompt) in enumerate(tqdm(list(image_filename_to_caption.items()), desc="Generating images")):
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n\n{'='*80}")
        print(f"Processing image {i+1}/{len(image_filename_to_caption)}: {filename} (Prompt: {prompt[:50]}...)")
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
            print(f"Generated image {i+1}/{len(image_filename_to_caption)}: {filename}")
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
    
    if not args.skip_metrics:
        # Create directory for resized images (needed for FID and PSNR)
        resized_output_dir = os.path.join(output_dir, "resized")
        os.makedirs(resized_output_dir, exist_ok=True)
        
        # Resize images for metrics calculation
        try:
            print("\n--- Resizing Images for Metrics Calculation ---")
            resize_images(output_dir, resized_output_dir, image_dimensions)
        except Exception as e:
            print(f"Error resizing images: {e}")
        
        # Calculate FID score
        try:
            print("\n--- Calculating FID Score ---")
            coco_val_dir = os.path.join(coco_dir, "val2017")
            fid_score = calculate_fid(output_dir, resized_output_dir, coco_val_dir)
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
            # For COCO dataset, we need to select a subset of filenames for LPIPS calculation
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
            
            # Manually resize original COCO images to match generated images for LPIPS calculation
            original_dir = os.path.join(coco_dir, "val2017")
            for filename in tqdm(selected_filenames, desc="Resizing original images for LPIPS"):
                original_path = os.path.join(original_dir, filename)
                resized_original_path = os.path.join(resized_original_dir, filename)
                
                if os.path.exists(original_path):
                    try:
                        # Get the size of the resized generated image for consistency
                        generated_img_path = os.path.join(resized_output_dir, filename)
                        generated_img = Image.open(generated_img_path)
                        target_size = generated_img.size
                        
                        # Resize the original image to match the generated image
                        original_img = Image.open(original_path).convert("RGB")
                        resized_original_img = original_img.resize(target_size, Image.LANCZOS)
                        resized_original_img.save(resized_original_path)
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
            # For COCO dataset, we use resized generated images
            original_dir = os.path.join(coco_dir, "val2017")
            generated_filenames = [f for f in os.listdir(resized_output_dir) if os.path.isfile(os.path.join(resized_output_dir, f))
                                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            # Use the metrics_subset parameter to limit the number of images for metrics
            subset_size = min(args.metrics_subset, len(generated_filenames))
            psnr_score = calculate_psnr_resized(original_dir, resized_output_dir, generated_filenames[:subset_size])
            metrics_results["psnr"] = psnr_score
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
    
    # Save metadata to JSON file
    metadata_file = os.path.join(output_dir, "pruning_metadata.json")
    try:
        # Add metrics to metadata
        combined_metadata = {
            "model_info": {
                "pruning_method": args.pruning_method,
                "pruning_amount": args.pruning_amount,
                "model_size_mb": pruned_size,
                "sparsity": sparsity
            },
            "generation_metadata": generation_metadata,
            "metrics": metrics_results
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(combined_metadata, f, ensure_ascii=False, indent=4)
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
            overall_avg_vram = sum(all_vram_data) / len(all_vram_data)
            print(f"Average VRAM usage across all {len(all_vram_data)} images: {overall_avg_vram:.2f} GB")
    
    # Save summary report
    with open(os.path.join(output_dir, "pruning_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Pruning Summary ===\n\n")
        f.write(f"Processed {len(image_filename_to_caption)} COCO captions\n")
        f.write(f"Inference steps: {args.steps}\n")
        f.write(f"Pruning method: {args.pruning_method}\n")
        f.write(f"Pruning amount: {args.pruning_amount*100:.1f}%\n")
        f.write(f"Pruned model size: {pruned_size:.2f} MB\n")
        f.write(f"Model sparsity: {sparsity:.2f}%\n")
        if successful_generations:
            f.write(f"Successfully generated {len(successful_generations)}/{len(image_filename_to_caption)} images\n")
            f.write(f"Average generation time: {avg_time:.2f} seconds per image\n")
            f.write(f"Total generation time: {sum(successful_generations):.2f} seconds\n")
            # Add VRAM usage to summary
            all_vram_data = [meta.get("average_vram_gb") for meta in generation_metadata if meta.get("average_vram_gb") is not None]
            if all_vram_data:
                overall_avg_vram = sum(all_vram_data) / len(all_vram_data)
                f.write(f"Average VRAM usage across all {len(all_vram_data)} images: {overall_avg_vram:.2f} GB\n")
        
        # Add metrics results to summary
        if metrics_results:
            f.write("\n=== Image Quality Metrics ===\n")
            for metric_name, metric_value in metrics_results.items():
                if metric_value is not None:
                    f.write(f"{metric_name.upper()}: {metric_value:.4f}\n")
    
    print(f"Completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
