#!/usr/bin/env python3

import argparse
import gc
import os
from PIL import Image
import torch
from tqdm import tqdm

from diffusers import FluxPipeline
from huggingface_hub import login
# Fix import paths to be relative to the project root
import sys
import os
# Add the project root to the path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import calculate_clip_score, calculate_fid, calculate_lpips, calculate_psnr_resized, compute_image_reward
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision
# Import preprocessing from current directory since we're already in the quantization module
# Use explicit import path to make sure it works both when imported and when run directly
import os
import sys
# Get the absolute path to the quantization directory
quantization_dir = os.path.dirname(os.path.abspath(__file__))
if quantization_dir not in sys.path:
    sys.path.append(quantization_dir)
from quantization.preprocessing_coco import preprocessing_coco
from resizing_image import resize_images
from shared.resources_monitor import generate_image_and_monitor, write_generation_metadata_to_file


def setup_memory_optimizations():
    """Apply memory optimizations to avoid CUDA OOM errors"""
    # PyTorch memory optimizations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_model(precision_override=None):
    """Load the model with specified precision or auto-detect"""
    try:
        # Auto-detect precision if not specified
        precision = precision_override if precision_override else get_precision()
        print(f"Using precision: {precision}")
        
        # Load transformer model
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"mit-han-lab/svdq-{precision}-flux.1-schnell", 
            offload=True
        )
        return transformer, precision
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def create_pipeline(transformer):
    """Create pipeline from transformer model"""
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", 
        transformer=transformer, 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    pipeline.enable_model_cpu_offload()
    return pipeline


def get_model_size(model):
    """Calculate the model size in MB"""
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)
    return model_size_mb


def main(args=None):
    # Setup argument parser if args not provided
    if args is None:
        parser = argparse.ArgumentParser(description="Quantization with COCO dataset")
        # parser.add_argument("--bits", type=int, default=8, 
        #                     choices=[8, 4, 2],
        #                     help="Bit precision for quantization")
        parser.add_argument("--quant_method", type=str, default="dynamic", 
                            choices=["static", "dynamic", "aware_training"],
                            help="Quantization method to use")
        parser.add_argument("--precision", type=str, default=None, 
                            help="Precision to use (default is auto-detected)")
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
    
    # Apply memory optimizations
    setup_memory_optimizations()

    # Replace 'YOUR_TOKEN' with your actual Hugging Face token
    login(token="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")

    # Define paths
    coco_dir = "coco"
    annotations_dir = os.path.join(coco_dir, "annotations")
    val2017_dir = os.path.join(coco_dir, "val2017")

    # Process COCO dataset
    print("\n=== Loading COCO Captions ===")
    image_filename_to_caption, image_dimensions, image_id_to_dimensions = preprocessing_coco(annotations_dir)

    # Create output directories
    generation_output_dir = "quantization/quant_outputs/coco"
    os.makedirs(generation_output_dir, exist_ok=True)
    print(f"Created directory for generated images: {generation_output_dir}")

    resized_output_dir = os.path.join(generation_output_dir, "resized")
    os.makedirs(resized_output_dir, exist_ok=True)
    print(f"Created directory for resized generated images: {resized_output_dir}")

    # 1. Load the model with quantization
    print("\n=== Loading Quantized Model ===")
    try:
        transformer, precision = load_model(args.precision)
        quantized_size = get_model_size(transformer)
        print(f"Quantized model size: {quantized_size:.2f} MB")
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        print("Trying again with default parameters...")
        try:
            # Fallback to default model
            model_path = "mit-han-lab/svdq-int4-flux.1-dev"
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(model_path, offload=True)
            precision = "int4"
            quantized_size = get_model_size(transformer)
            print(f"Loaded fallback model with size: {quantized_size:.2f} MB")
        except Exception as e2:
            print(f"Fatal error loading model: {e2}")
            return

    # Create pipeline with the quantized model
    print("\n=== Creating Pipeline ===")
    try:
        pipeline = create_pipeline(transformer)
        
        # Test the pipeline with a simple prompt to ensure it works
        print("Testing pipeline with a simple prompt...")
        test_image = pipeline(
            "a high quality photo of a motorcycle",
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
    
    # 2. Generate images for COCO captions
    print(f"\n=== Generating Images for COCO Captions ===")
    
    # Limit the number of images to generate based on the command-line argument
    num_images_to_generate = min(args.num_images, len(image_filename_to_caption))
    print(f"Will generate {num_images_to_generate} images")
    
    # Create a reverse mapping from filename to image_id
    image_filename_to_id = {v[2]: k for k, v in image_id_to_dimensions.items()}
    
    # Keep track of generation time
    generation_times = {}
    
    for i, (filename, prompt) in enumerate(tqdm(list(image_filename_to_caption.items())[:num_images_to_generate], desc="Generating images")):
        output_path = os.path.join(generation_output_dir, filename)
        
        print(f"\n\n{'='*80}")
        print(f"Processing image {i+1}/{num_images_to_generate}: {filename} (Prompt: {prompt[:50]}...)")
        print(f"Output will be saved to: {output_path}")
        print(f"{'='*80}")
        
        # Skip if the image already exists
        if os.path.exists(output_path):
            print(f"Skipping generation for {filename} (already exists)")
            continue
            
        try:
            # Generate image and monitor VRAM
            generation_time, metadata = generate_image_and_monitor(
                pipeline, prompt, output_path, filename, 
                num_inference_steps=args.steps, guidance_scale=args.guidance_scale
            )
            # Add the metadata to our list
            generation_metadata.append(metadata)
            # Add the generation time to the dictionary
            generation_times[filename] = generation_time
            print(f"Generated image {i+1}/{num_images_to_generate}: {filename}")
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

    # Save generation metadata
    metadata_file = os.path.join(generation_output_dir, "coco_quantization_metadata.json")
    write_generation_metadata_to_file(metadata_file)

    # Calculate and print average generation time
    successful_generations = [t for t in generation_times.values() if t > 0]
    if successful_generations:
        avg_time = sum(successful_generations) / len(successful_generations)
        print(f"Average generation time per image (for successful generations): {avg_time:.2f} seconds")

    # 3. Resize images for comparison
    try:
        print("\n=== Resizing Images ===")
        resize_images(generation_output_dir, resized_output_dir, image_dimensions)
        print("Image resizing complete.")
    except Exception as e:
        print(f"Error resizing images: {e}")

    # 4. Calculate Image Quality Metrics (if not skipped)
    metrics_results = {}
    
    if args.skip_metrics:
        print("\n=== Skipping Image Quality Metrics (--skip_metrics flag set) ===")
    else:
        print("\n=== Calculating Image Quality Metrics ===")
    
    if not args.skip_metrics:
        # Create directory for resized images (needed for FID and PSNR)
        resized_original_dir = os.path.join(generation_output_dir, "resized_original")
        os.makedirs(resized_original_dir, exist_ok=True)
        
        # Calculate FID score
        try:
            print("\n--- Calculating FID Score ---")
            fid_score = calculate_fid(generation_output_dir, resized_output_dir, val2017_dir)
            metrics_results["fid_score"] = fid_score
        except Exception as e:
            print(f"Error calculating FID: {e}")
        
        # Calculate CLIP Score
        try:
            print("\n--- Calculating CLIP Score ---")
            clip_score = calculate_clip_score(generation_output_dir, image_filename_to_caption)
            metrics_results["clip_score"] = clip_score
        except Exception as e:
            print(f"Error calculating CLIP Score: {e}")
        
        # Calculate ImageReward
        try:
            print("\n--- Calculating ImageReward ---")
            image_reward = compute_image_reward(generation_output_dir, image_filename_to_caption)
            metrics_results["image_reward"] = image_reward
        except Exception as e:
            print(f"Error calculating ImageReward: {e}")
        
        # Calculate LPIPS - we need original images to compare with generated images
        try:
            print("\n--- Calculating LPIPS ---")
            # For COCO dataset, we need to select a subset of filenames for LPIPS calculation
            # We should compare resized images to ensure dimensions match
            
            # Get list of generated filenames that have been resized
            generated_filenames = [f for f in os.listdir(resized_output_dir) 
                                if os.path.isfile(os.path.join(resized_output_dir, f)) 
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
            generated_filenames = [f for f in os.listdir(resized_output_dir) 
                                if os.path.isfile(os.path.join(resized_output_dir, f))
                                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            # Use the metrics_subset parameter to limit the number of images for metrics
            subset_size = min(args.metrics_subset, len(generated_filenames))
            psnr_score = calculate_psnr_resized(original_dir, resized_output_dir, generated_filenames[:subset_size])
            metrics_results["psnr"] = psnr_score
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
    
    # Calculate and print statistics
    if successful_generations:
        avg_time = sum(successful_generations) / len(successful_generations)
        print(f"\n=== Generation Statistics ===")
        print(f"Successfully generated {len(successful_generations)}/{num_images_to_generate} images")
        print(f"Average generation time: {avg_time:.2f} seconds per image")
        print(f"Total generation time: {sum(successful_generations):.2f} seconds")
        
        # Calculate average VRAM usage across all images
        all_vram_data = [meta.get("average_vram_gb") for meta in generation_metadata if meta.get("average_vram_gb") is not None]
        if all_vram_data:
            overall_avg_vram = sum(all_vram_data) / len(all_vram_data)
            print(f"Average VRAM usage across all {len(all_vram_data)} images: {overall_avg_vram:.2f} GB")
    
    # Save summary report
    summary_file = os.path.join(generation_output_dir, "quantization_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=== Quantization Summary ===\n\n")
        f.write(f"Processed {num_images_to_generate} COCO captions\n")
        f.write(f"Inference steps: {args.steps}\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n")
        f.write(f"Quantization: {precision}\n")
        f.write(f"Model size: {quantized_size:.2f} MB\n")
        if successful_generations:
            f.write(f"Successfully generated {len(successful_generations)}/{num_images_to_generate} images\n")
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
    
    print(f"Completed. Results saved to {generation_output_dir}")
    print(f"Summary report: {summary_file}")


if __name__ == "__main__":
    main()
