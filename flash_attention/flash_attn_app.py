#!/usr/bin/env python3
"""
Example script for using Flash Attention with Diffusion Models.

This script demonstrates how to use the Flash Attention optimization
together with other optimizations (quantization, pruning, KV caching)
for efficient image generation with diffusion models.
"""

import os
import sys
import argparse

from tqdm import tqdm
from PIL import Image
import torch
import time
import gc
import json
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from flash_attention.flash_attention_pipeline import FlashAttentionOptimizedPipeline
    from flash_attention.memory_utils import aggressive_memory_cleanup, get_memory_info, print_memory_info, optimize_inference_environment
    from dataset.flickr8k import process_flickr8k
    from shared.metrics import calculate_fid, compute_image_reward, calculate_clip_score, calculate_lpips, calculate_psnr_resized
    from shared.resources_monitor import generate_image_and_monitor, monitor_vram, write_generation_metadata_to_file
    from shared.cleanup import setup_memory_optimizations_pruning, optimize_for_inference
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configure PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Flash Attention Enhanced Diffusion Model Generation")
    
    # Model and optimization parameters
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-schnell",
                        help="Model path or HuggingFace repo ID")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "int8", "int4"],
                        help="Precision to use for quantization")
    parser.add_argument("--pruning", type=float, default=0.0,
                        help="Pruning amount (0.0 to 0.9)")
    parser.add_argument("--kv-cache", action="store_true", default=True,
                        help="Use KV caching for faster generation")
    parser.add_argument("--flash-attn", action="store_true", default=False,
                        help="Use Flash Attention for faster generation")
    parser.add_argument("--use-8bit", action="store_true", default=False,
                        help="Use 8-bit quantization for model loading")
    parser.add_argument("--use-4bit", action="store_true", default=False,
                        help="Use 4-bit quantization for model loading")
    parser.add_argument("--use-fp16", action="store_true", default=True,
                        help="Use float16 precision")
    parser.add_argument("--device-map", type=str, default="auto",
                        help="Device mapping strategy")
    
    # Generation parameters
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt for image generation")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--height", type=int, default=512,
                        help="Height of generated image")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of generated image")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, choices=["flickr8k", "coco", "none"], default="none",
                        help="Dataset to use for generation (flickr8k, coco, or none for single prompt)")
    parser.add_argument("--num-images", type=int, default=10,
                        help="Number of images to generate from dataset")
    parser.add_argument("--prompt", type=str, default="a beautiful landscape with mountains and a lake",
                        help="Single prompt for image generation (used when dataset is 'none')")
    
    # Benchmark parameters
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmarks comparing Flash Attention vs standard attention")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials for benchmarking")
    
    # Metrics parameters
    parser.add_argument("--skip-metrics", action="store_true",
                        help="Skip calculation of image quality metrics")
    parser.add_argument("--metrics-subset", type=int, default=500,
                        help="Number of images to use for metrics calculation")
    parser.add_argument("--monitor-vram", action="store_true",
                        help="Monitor VRAM usage during image generation")
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="flash_attention_outputs",
                        help="Base output directory for generated images and metrics")
    
    return parser.parse_args()

def generate_dataset_images(pipeline, args):
    """Generate images from a dataset with metrics calculation and VRAM monitoring"""
    # Clean up memory before processing
    torch.cuda.empty_cache()
    gc.collect()
    
    # Process dataset
    captions = {}
    image_dimensions = {}
    original_images_dir = None
    
    try:
        if args.dataset == "flickr8k":
            print("Processing Flickr8k dataset...")
            flickr8k_dir = "flickr8k"
            images_dir = os.path.join(flickr8k_dir, "Images")
            captions_file = os.path.join(flickr8k_dir, "captions.txt")
            original_images_dir = images_dir  # Store for metrics calculation
            
            print("\n=== Loading Flickr8k Captions ===")
            # Process flickr8k dataset (now returns just captions dictionary)
            captions, image_dimensions = process_flickr8k(images_dir, captions_file)
            print(f"\n=== Generating Images for {len(captions)} Flickr8k Captions ===")
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return

    # Create output directory with timestamp and optimization details
    timestamp = datetime.now().strftime('%Y%m%d')
    optimization_details = []
    if args.flash_attn:
        optimization_details.append("flash")
    if args.kv_cache:
        optimization_details.append("kvcache")
    if args.precision != "fp16":
        optimization_details.append(f"quant_{args.precision}")
    if args.pruning > 0:
        optimization_details.append(f"prune_{int(args.pruning*100)}")
    
    optim_suffix = "_".join(optimization_details) if optimization_details else "baseline"
    
    output_dir = os.path.join(
        args.output_dir,
        f"{args.dataset}_{optim_suffix}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Generating {args.num_images} images from {args.dataset} dataset")
    print(f"Flash Attention: {args.flash_attn}, KV Cache: {args.kv_cache}")
    print(f"VRAM Monitoring: {args.monitor_vram}")
    print(f"Image size: {args.height}x{args.width}, Steps: {args.steps}")
    
    # Limit the number of images to generate
    if args.num_images < len(captions):
        # Get the first num_images captions
        image_keys = list(captions.keys())[:args.num_images]
        filtered_captions = {k: captions[k] for k in image_keys}
        print(f"Selected the first {args.num_images} captions from the dataset")
    else:
        filtered_captions = captions
        print(f"Using all {len(captions)} captions from the dataset")
    
    # Track generation metadata
    generation_times = {}
    all_metadata = []
    successful_generations = 0
    
    try:
        # Generate images one by one
        print(f"\nStarting image generation...")
        
        for i, (filename, caption) in enumerate(filtered_captions.items(), 1):
            print(f"\n[{i}/{len(filtered_captions)}] Processing: {filename}")
            print(f"Caption: {caption[:100]}...")
            
            # Prepare output path
            output_filename = f"{filename}"
            if not output_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                output_filename += '.png'
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                # Generate image with optional VRAM monitoring
                if args.monitor_vram:
                    generation_time, metadata = monitor_generation_with_vram(
                        pipeline, caption, output_path, filename, args
                    )
                else:
                    # Standard generation without VRAM monitoring
                    start_time = time.time()
                    
                    # Generate using the pipeline's generate_image method
                    if hasattr(pipeline, 'generate_image'):
                        result = pipeline.generate_image(
                            prompt=caption,
                            num_inference_steps=args.steps,
                            guidance_scale=args.guidance,
                            use_cache=args.kv_cache,
                            use_flash=args.flash_attn,
                            height=args.height,
                            width=args.width
                        )
                        
                        # Handle the return value (could be image or tuple)
                        if isinstance(result, tuple):
                            image, pipeline_generation_time = result
                        else:
                            image = result
                            pipeline_generation_time = time.time() - start_time
                    else:
                        # Fallback to direct pipeline call
                        pipeline_result = pipeline.pipeline(
                            prompt=caption,
                            num_inference_steps=args.steps,
                            guidance_scale=args.guidance,
                            height=args.height,
                            width=args.width
                        )
                        image = pipeline_result.images[0] if hasattr(pipeline_result, 'images') else pipeline_result
                        pipeline_generation_time = time.time() - start_time
                    
                    generation_time = pipeline_generation_time
                    
                    # Save image
                    if hasattr(image, 'save'):
                        image.save(output_path)
                    else:
                        # Handle different image types
                        from PIL import Image
                        if not isinstance(image, Image.Image):
                            image = Image.fromarray(image)
                        image.save(output_path)
                    
                    metadata = {
                        "generation_time": generation_time,
                        "vram_monitoring": False
                    }
                
                if generation_time > 0:
                    generation_times[filename] = generation_time
                    successful_generations += 1
                    print(f"✅ Generated in {generation_time:.2f}s")
                    
                    # Add comprehensive metadata
                    full_metadata = {
                        "filename": filename,
                        "output_path": output_path,
                        "caption": caption,
                        "generation_time": generation_time,
                        "image_index": i,
                        "optimization_settings": {
                            "flash_attention": args.flash_attn,
                            "kv_cache": args.kv_cache,
                            "precision": args.precision,
                            "pruning": args.pruning
                        },
                        "generation_settings": {
                            "steps": args.steps,
                            "guidance_scale": args.guidance,
                            "height": args.height,
                            "width": args.width
                        }
                    }
                    full_metadata.update(metadata)  # Add VRAM data if available
                    all_metadata.append(full_metadata)
                    
                else:
                    print(f"❌ Generation failed")
                    
            except torch.cuda.OutOfMemoryError as e:
                print(f"❌ CUDA Out of Memory for {filename}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue
                
            except Exception as e:
                print(f"❌ Error generating {filename}: {e}")
                continue
            
            # Periodic memory cleanup
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
        # Save generation metadata
        metadata_file = os.path.join(output_dir, "generation_metadata.json")
        write_generation_metadata_to_file(metadata_file, all_metadata)
        
        # Calculate generation statistics
        successful_generations_times = [t for t in generation_times.values() if t > 0]
        if successful_generations_times:
            avg_time = sum(successful_generations_times) / len(successful_generations_times)
            print(f"\n=== Generation Statistics ===")
            print(f"Successfully generated {len(successful_generations_times)}/{len(filtered_captions)} images")
            print(f"Average generation time: {avg_time:.2f} seconds per image")
            print(f"Total generation time: {sum(successful_generations_times):.2f} seconds")
        
        # Calculate and save metrics if not skipped
        if not args.skip_metrics and successful_generations > 0:
            print(f"\n=== Calculating Metrics for {successful_generations} Generated Images ===")
            
            metrics_results = calculate_all_metrics(
                generated_dir=output_dir,
                original_dir=original_images_dir,
                captions_dict=filtered_captions,
                image_dimensions=image_dimensions,
                args=args
            )
            
            # Save metrics results
            # metrics_file = os.path.join(output_dir, "metrics_results.json")
            # try:
            #     import json
            #     with open(metrics_file, 'w') as f:
            #         json.dump(metrics_results, f, indent=2)
            #     print(f"Metrics results saved to: {metrics_file}")
            # except Exception as e:
            #     print(f"Error saving metrics: {e}")
        
        # Generate summary report
        summary_file = os.path.join(output_dir, "generation_summary.txt")
        try:
            with open(summary_file, 'w') as f:
                f.write(f"Flash Attention Enhanced Image Generation Summary\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Dataset: {args.dataset}\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Optimization Settings:\n")
                f.write(f"  - Flash Attention: {args.flash_attn}\n")
                f.write(f"  - KV Cache: {args.kv_cache}\n")
                f.write(f"  - Precision: {args.precision}\n")
                f.write(f"  - Pruning: {args.pruning}\n")
                f.write(f"Generation Settings:\n")
                f.write(f"  - Steps: {args.steps}\n")
                f.write(f"  - Guidance Scale: {args.guidance}\n")
                f.write(f"  - Image Size: {args.height}x{args.width}\n")
                f.write(f"Results:\n")
                f.write(f"  - Total Attempts: {len(filtered_captions)}\n")
                f.write(f"  - Successful Generations: {successful_generations}\n")
                f.write(f"  - Success Rate: {successful_generations/len(filtered_captions)*100:.1f}%\n")
                
                # Use improved generation time calculation
                successful_generations_times = [t for t in generation_times.values() if t > 0]
                if successful_generations_times:
                    avg_time = sum(successful_generations_times) / len(successful_generations_times)
                    f.write(f"  - Average Generation Time: {avg_time:.2f}s\n")
                    f.write(f"  - Min Generation Time: {min(successful_generations_times):.2f}s\n")
                    f.write(f"  - Max Generation Time: {max(successful_generations_times):.2f}s\n")
                    f.write(f"  - Total Generation Time: {sum(successful_generations_times):.2f}s\n")
                
                if not args.skip_metrics and successful_generations > 0:
                    f.write(f"\nMetrics Results:\n")
                    for metric_name, value in metrics_results.items():
                        if value is not None:
                            f.write(f"  - {metric_name.upper().replace('_', ' ')}: {value:.4f}\n")
                        else:
                            f.write(f"  - {metric_name.upper().replace('_', ' ')}: N/A\n")
                
            print(f"Summary report saved to: {summary_file}")
        except Exception as e:
            print(f"Error saving summary: {e}")
        
        print(f"\n🎉 Generation completed!")
        
        # Use improved generation time calculation for final summary
        successful_generations_times = [t for t in generation_times.values() if t > 0]
        print(f"✅ Successfully generated: {len(successful_generations_times)}/{len(filtered_captions)} images")
        if successful_generations_times:
            avg_time = sum(successful_generations_times) / len(successful_generations_times)
            print(f"⏱️ Average generation time: {avg_time:.2f}s per image")
            print(f"⏱️ Total generation time: {sum(successful_generations_times):.2f}s")
        print(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during dataset generation: {e}")
        torch.cuda.empty_cache()
        gc.collect()

def run_benchmark(pipeline, args):
    """Run benchmark comparing Flash Attention vs standard attention"""
    print("\n" + "=" * 50)
    print("RUNNING BENCHMARK: Flash Attention vs Standard Attention")
    print("=" * 50)
    
    # Set up benchmark configurations
    configs = [
        {"name": "Standard Attention + KV Cache", "flash": False, "kv_cache": True},
        {"name": "Flash Attention + KV Cache", "flash": True, "kv_cache": True},
        {"name": "Standard Attention Only", "flash": False, "kv_cache": False},
        {"name": "Flash Attention Only", "flash": True, "kv_cache": False},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        times = []
        memories = []
        
        for trial in range(args.trials):
            print(f"  Trial {trial + 1}/{args.trials}...")
            
            # Clear caches
            if hasattr(pipeline, 'flash_transformer') and hasattr(pipeline.flash_transformer, 'clear_cache'):
                pipeline.flash_transformer.clear_cache()
            
            torch.cuda.empty_cache()
            
            # Create a temporary output path for benchmark
            benchmark_output = f"/tmp/benchmark_image_{trial}.png"
            
            # Generate image with proper method call
            try:
                result = pipeline.generate_image(
                    prompt=args.prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    use_cache=config["kv_cache"],
                    use_flash=config["flash"],
                    height=args.height,
                    width=args.width,
                    seed=args.seed if args.seed else trial + 1000
                )
                
                # Handle the return value (could be image or tuple)
                if isinstance(result, tuple):
                    image, generation_time = result
                else:
                    image = result
                    generation_time = -1  # Unknown time
                
                # Get memory usage (rough estimate)
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                    torch.cuda.reset_peak_memory_stats()
                else:
                    peak_memory = 0
                
                # Handle different image return types
                if hasattr(image, 'images'):
                    image = image.images[0]
                
                # Save benchmark image temporarily
                image.save(benchmark_output)
                
                times.append(generation_time)
                memories.append(peak_memory)
                
                # Clean up benchmark image
                if os.path.exists(benchmark_output):
                    os.remove(benchmark_output)
                
            except Exception as e:
                print(f"    Error in trial {trial + 1}: {e}")
                times.append(float('inf'))  # Indicate failure
                memories.append(0)
            
            # Add a small delay between trials
            time.sleep(1)
        
        # Compute statistics (excluding failed trials)
        valid_times = [t for t in times if t != float('inf')]
        valid_memories = [m for m in memories if m != 0]
        
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
        else:
            avg_time = float('inf')
            
        if valid_memories:
            avg_memory = sum(valid_memories) / len(valid_memories)
        else:
            avg_memory = 0
        
        results[config["name"]] = {
            "avg_time": avg_time,
            "avg_memory": avg_memory,
            "times": times,
            "memories": memories,
            "successful_trials": len(valid_times),
            "total_trials": args.trials
        }
        
        if avg_time != float('inf'):
            print(f"  Average time: {avg_time:.2f} seconds ({len(valid_times)}/{args.trials} successful)")
        else:
            print(f"  All trials failed")
            
        if avg_memory > 0:
            print(f"  Average memory: {avg_memory:.2f} MB")
    
    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    # Find baseline for comparison (first successful config)
    baseline_time = None
    baseline_memory = None
    baseline_name = None
    
    for name, data in results.items():
        if data["avg_time"] != float('inf'):
            baseline_time = data["avg_time"]
            baseline_memory = data["avg_memory"]
            baseline_name = name
            break
    
    if baseline_time is None:
        print("No successful benchmark runs to compare")
        return results
    
    print(f"Using '{baseline_name}' as baseline")
    
    for name, data in results.items():
        if data["avg_time"] != float('inf'):
            time_speedup = baseline_time / data["avg_time"] if data["avg_time"] > 0 else 0
            if baseline_memory > 0 and data["avg_memory"] > 0:
                memory_change = (data["avg_memory"] - baseline_memory) / baseline_memory * 100
            else:
                memory_change = 0
            
            print(f"\n{name}:")
            print(f"  Time: {data['avg_time']:.2f} seconds ({time_speedup:.2f}x vs baseline)")
            print(f"  Memory: {data['avg_memory']:.2f} MB ({memory_change:+.1f}% vs baseline)")
            print(f"  Success Rate: {data['successful_trials']}/{data['total_trials']}")
        else:
            print(f"\n{name}: All trials failed")
    
    # Save benchmark results
    benchmark_dir = os.path.join(args.output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Convert inf values to string for JSON serialization
    serializable_results = {}
    for name, data in results.items():
        serializable_data = data.copy()
        if serializable_data["avg_time"] == float('inf'):
            serializable_data["avg_time"] = "failed"
        serializable_data["times"] = [t if t != float('inf') else "failed" for t in serializable_data["times"]]
        serializable_results[name] = serializable_data
    
    try:
        import json
        with open(os.path.join(benchmark_dir, "benchmark_results.json"), "w") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nBenchmark results saved to: {benchmark_dir}")
    except Exception as e:
        print(f"Error saving benchmark results: {e}")
    
    return results

def calculate_all_metrics(generated_dir, original_dir, captions_dict, image_dimensions, args):
    """
    Calculate all available metrics for generated images
    
    Args:
        generated_dir: Directory containing generated images
        original_dir: Directory containing original images (for comparison metrics)
        captions_dict: Dictionary mapping filenames to captions
        image_dimensions: Dictionary mapping filenames to their original dimensions (width, height)
        args: Command line arguments
        
    Returns:
        Dictionary containing all calculated metrics
    """
    print("\n" + "=" * 50)
    print("CALCULATING IMAGE QUALITY METRICS")
    print("=" * 50)
    
    metrics_results = {}
    
    # Get list of generated images
    generated_files = [f for f in os.listdir(generated_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Found {len(generated_files)} generated images for metrics calculation")
    
    if not generated_files:
        print("No generated images found for metrics calculation")
        return metrics_results
    
    # Limit to subset if specified
    if args.metrics_subset and len(generated_files) > args.metrics_subset:
        generated_files = generated_files[:args.metrics_subset]
        print(f"Limited metrics calculation to {len(generated_files)} images")
    
    try:
        # 1. Calculate CLIP Score
        print("\n1. Calculating CLIP Score...")
        clip_score = calculate_clip_score(generated_dir, captions_dict)
        metrics_results["clip_score"] = clip_score
        
    except Exception as e:
        print(f"Error calculating CLIP Score: {e}")
        metrics_results["clip_score"] = None
    
    try:
        # 2. Calculate Image Reward
        print("\n2. Calculating Image Reward...")
        image_reward = compute_image_reward(generated_dir, captions_dict)
        metrics_results["image_reward"] = image_reward
        
    except Exception as e:
        print(f"Error calculating Image Reward: {e}")
        metrics_results["image_reward"] = None
    
    # 3. Calculate FID Score (if original images available)
    resized_generated_dir = os.path.join(generated_dir, "resized")
    if original_dir and os.path.exists(original_dir):
        try:
            print("\n3. Calculating FID Score...")
            # First, create resized versions for fair comparison
            from shared.resizing_image import resize_images
            os.makedirs(resized_generated_dir, exist_ok=True)
            
            # Resize generated images to match original dimensions
            resize_images(generated_dir, resized_generated_dir, image_dimensions)
            
            fid_score = calculate_fid(generated_dir, resized_generated_dir, original_dir)
            metrics_results["fid_score"] = fid_score
            
        except Exception as e:
            print(f"Error calculating FID Score: {e}")
            metrics_results["fid_score"] = None
    else:
        print("\n3. Skipping FID Score (no original images directory provided)")
        metrics_results["fid_score"] = None
    
    # 4. Calculate LPIPS (if original images available)
    if original_dir and os.path.exists(original_dir):
        try:
            print("\n4. Calculating LPIPS...")
            # Create a directory for resized original images
            resized_original_dir = os.path.join(generated_dir, "resized_original")
            os.makedirs(resized_original_dir, exist_ok=True)
            
            # Get list of generated filenames that have been resized
            generated_filenames = [f for f in os.listdir(resized_generated_dir) if os.path.isfile(os.path.join(resized_generated_dir, f)) 
                                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            # Use the metrics_subset parameter to limit the number of images for metrics
            subset_size = min(args.metrics_subset, len(generated_filenames))
            selected_filenames = generated_filenames[:subset_size]
            
            # Manually resize original images dataset to match generated images for LPIPS calculation
            for filename in tqdm(selected_filenames, desc="Resizing original images for LPIPS"):
                original_path = os.path.join(original_dir, filename)
                resized_original_path = os.path.join(resized_original_dir, filename)
                
                if os.path.exists(original_path):
                    try:
                        # Get the size of the resized generated image for consistency
                        generated_img_path = os.path.join(resized_generated_dir, filename)
                        generated_img = Image.open(generated_img_path)
                        target_size = generated_img.size
                        
                        # Resize the original image to match the generated image
                        original_img = Image.open(original_path).convert("RGB")
                        resized_original_img = original_img.resize(target_size, Image.LANCZOS)
                        resized_original_img.save(resized_original_path)
                    except Exception as e:
                        print(f"Error resizing original image {filename}: {e}")
            
            # Now calculate LPIPS using the resized original and generated images
            lpips_score = calculate_lpips(resized_original_dir, resized_generated_dir, selected_filenames)
            metrics_results["lpips_score"] = lpips_score
            
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")
            metrics_results["lpips_score"] = None
    else:
        print("\n4. Skipping LPIPS (no original images directory provided)")
        metrics_results["lpips_score"] = None
    
    # 5. Calculate PSNR (if original images available)
    if original_dir and os.path.exists(original_dir):
        try:
            print("\n5. Calculating PSNR...")
            # Use resized images for fair comparison
            
            if os.path.exists(resized_generated_dir):
                psnr_score = calculate_psnr_resized(original_dir, resized_generated_dir, generated_files)
            else:
                psnr_score = calculate_psnr_resized(original_dir, generated_dir, generated_files)
            metrics_results["psnr_score"] = psnr_score
            
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            metrics_results["psnr_score"] = None
    else:
        print("\n5. Skipping PSNR (no original images directory provided)")
        metrics_results["psnr_score"] = None
    
    # Print summary of all metrics
    print("\n" + "=" * 50)
    print("METRICS SUMMARY")
    print("=" * 50)
    for metric_name, metric_value in metrics_results.items():
        if metric_value is not None:
            print(f"{metric_name.upper().replace('_', ' ')}: {metric_value:.4f}")
        else:
            print(f"{metric_name.upper().replace('_', ' ')}: N/A")
    
    return metrics_results

def monitor_generation_with_vram(pipeline, prompt, output_path, filename, args):
    """
    Generate image with VRAM monitoring for FlashAttentionOptimizedPipeline
    
    Args:
        pipeline: The FlashAttentionOptimizedPipeline instance
        prompt: Text prompt for generation
        output_path: Path to save the generated image
        filename: Original filename for metadata
        args: Command line arguments
        
    Returns:
        Tuple of (generation_time, metadata_dict)
    """
    if not args.monitor_vram:
        # If VRAM monitoring is disabled, use standard generation
        try:
            result = pipeline.generate_image(
                prompt=prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                use_cache=args.kv_cache,
                use_flash=args.flash_attn,
                height=args.height,
                width=args.width
            )
            
            # Handle the return value (could be image or tuple)
            if isinstance(result, tuple):
                image, generation_time = result
            else:
                image = result
                generation_time = -1  # Unknown time if not returned
            
            # Save image
            if hasattr(image, 'images'):
                image = image.images[0]
            image.save(output_path)
            
            return generation_time, {
                "generation_time": generation_time,
                "vram_monitoring": False
            }
        except Exception as e:
            print(f"Error during generation: {e}")
            return -1, {"error": str(e)}
    
    # Use VRAM monitoring with proper pipeline call
    from shared.resources_monitor import vram_samples, stop_monitoring, monitor_vram
    import threading
    
    # Reset VRAM samples and stop event for this run
    vram_samples.clear()
    stop_monitoring.clear()

    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_vram)
    monitor_thread.start()

    # Run the image generation function
    generation_time = -1
    metadata = {}
    
    try:
        start_time = time.time()
        
        # Use the pipeline's generate_image method
        result = pipeline.generate_image(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            use_cache=args.kv_cache,
            use_flash=args.flash_attn,
            height=args.height,
            width=args.width
        )
        
        # Handle the return value (could be image or tuple)
        if isinstance(result, tuple):
            image, pipeline_generation_time = result
            generation_time = pipeline_generation_time
        else:
            image = result
            end_time = time.time()
            generation_time = end_time - start_time

        # Handle different image return types
        if hasattr(image, 'images'):
            image = image.images[0]
        
        # Save the generated image
        image.save(output_path)
        print(f"\n✅ Image saved to: {output_path}")
        
        # Create metadata
        metadata = {
            "generated_image_path": output_path,
            "original_filename": filename,
            "caption_used": prompt,
            "generation_time": generation_time,
            "guidance_scale": args.guidance,
            "num_steps": args.steps,
            "vram_monitoring": True
        }
        
        print(f"Image generation completed in {generation_time:.2f} seconds.")

    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        generation_time = -1  # Indicate failure
        metadata = {"error": str(e)}
    finally:
        # Stop the monitoring thread
        stop_monitoring.set()
        monitor_thread.join()

    # Analyze the collected VRAM data
    if vram_samples and generation_time > 0:
        avg_vram = sum(vram_samples) / len(vram_samples)
        peak_vram = max(vram_samples)

        print(f"\n--- VRAM Usage Statistics ---")
        print(f"Average VRAM used: {avg_vram:.2f} GB")
        print(f"Peak VRAM used: {peak_vram:.2f} GB")
        print(f"Number of samples: {len(vram_samples)}")
        
        metadata["average_vram_gb"] = avg_vram
        metadata["peak_vram_gb"] = peak_vram
        metadata["vram_samples_count"] = len(vram_samples)
    else:
        print("No VRAM data was collected.")
        metadata["average_vram_gb"] = None
        metadata["peak_vram_gb"] = None

    return generation_time, metadata

def main():
    """Main function"""
    args = parse_arguments()
    
    print("\n" + "=" * 50)
    print("FLASH ATTENTION ENHANCED DIFFUSION MODEL")
    print("=" * 50)
    
    # Set up environment for optimal memory usage
    print("\nSetting up optimized environment for inference...")
    optimize_inference_environment()
    
    # Apply standard memory optimizations
    print("\nApplying memory optimization strategies...")
    setup_memory_optimizations_pruning()
    
    # Perform aggressive memory cleanup
    print("\nPerforming initial memory cleanup...")
    memory_info = aggressive_memory_cleanup()
    print_memory_info(memory_info)
    
    # Display configuration
    print("\n" + "-" * 50)
    print("CONFIGURATION SETTINGS:")
    print("-" * 50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Precision: {args.precision}")
    print(f"Pruning: {args.pruning}")
    print(f"KV Cache: {args.kv_cache}")
    print(f"Flash Attention: {args.flash_attn}")
    print(f"VRAM Monitoring: {args.monitor_vram}")
    print(f"Skip Metrics: {args.skip_metrics}")
    print(f"Using FP16: {args.use_fp16}")
    print(f"Device map: {args.device_map}")
    print("-" * 50)
    
    # Initialize pipeline with memory-efficient settings
    try:
        print("\nLoading model with optimized memory settings...")
        
        # Another memory cleanup right before model loading
        aggressive_memory_cleanup()
        
        pipeline = FlashAttentionOptimizedPipeline(
            model_path=args.model,
            precision=args.precision,
            pruning_amount=args.pruning,
            use_kv_cache=args.kv_cache,
            use_flash_attention=args.flash_attn,
            device_map=args.device_map
        )
        
        # Check memory status after loading
        print("\nMemory status after model loading:")
        print_memory_info()
        
        # Apply inference optimizations if available
        if hasattr(pipeline, 'pipeline') and pipeline.pipeline is not None:
            print("\nApplying inference optimizations to model...")
            
            # First, memory cleanup before optimizations
            aggressive_memory_cleanup()
            
            # Optimize UNet (most memory-intensive component)
            if hasattr(pipeline.pipeline, 'unet'):
                print("Optimizing UNet for inference...")
                pipeline.pipeline.unet = optimize_for_inference(pipeline.pipeline.unet)
                print("Applied inference optimizations to UNet")
                
                # Clean up after UNet optimization
                aggressive_memory_cleanup()
            
            # Apply to other components if available
            for component_name in ['vae', 'text_encoder', 'text_encoder_2']:
                if hasattr(pipeline.pipeline, component_name):
                    component = getattr(pipeline.pipeline, component_name)
                    if component is not None:
                        print(f"Optimizing {component_name} for inference...")
                        setattr(pipeline.pipeline, component_name, optimize_for_inference(component))
                        print(f"Applied inference optimizations to {component_name}")
                        
                        # Clean up after each component optimization
                        aggressive_memory_cleanup()
                        
            print("\nAll components optimized for inference")
            print_memory_info()
    
    except torch.cuda.OutOfMemoryError as e:
        print(f"\nCUDA Out of Memory while loading model: {e}")
        
        # Enhanced memory cleanup
        aggressive_memory_cleanup()
        print_memory_info()
        
        # Try with 4-bit quantization
        
        print("\nAttempting model load with 4-bit quantization...")
        pipeline = FlashAttentionOptimizedPipeline(
            model_path=args.model,
            precision="int4",
            pruning_amount=max(0.3, args.pruning),  # Use at least 30% pruning
            use_kv_cache=args.kv_cache,
            use_flash_attention=True,
            device_map="auto"
        )
        print("\nSuccessfully loaded model with 4-bit quantization")
            
    except Exception as e:
        print(f"\nError initializing pipeline: {e}")
        sys.exit(1)
    
    # Clean up memory before running tasks
    print("\nPreparing for inference tasks...")
    aggressive_memory_cleanup()
    print_memory_info()
    
    # Run benchmark if requested
    if args.benchmark:
        print("\nRunning benchmark mode...")
        run_benchmark(pipeline, args)
        # Final cleanup
        aggressive_memory_cleanup()
        return
    
    # Generate images
    try:
        if args.dataset != "none":
            print(f"\nGenerating images using {args.dataset} dataset...")
            generate_dataset_images(pipeline, args)
        else:
            print(f"\nGenerating single image with prompt: '{args.prompt}'")
            
            # Create output directory for single image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(args.output_dir, f"single_image_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, "generated_image.png")
            
            # Generate image with optional VRAM monitoring
            if args.monitor_vram:
                generation_time, metadata = monitor_generation_with_vram(
                    pipeline, args.prompt, output_path, "single_image", args
                )
            else:
                start_time = time.time()
                if hasattr(pipeline, 'generate_image'):
                    result = pipeline.generate_image(
                        prompt=args.prompt,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        use_cache=args.kv_cache,
                        use_flash=args.flash_attn,
                        height=args.height,
                        width=args.width
                    )
                    
                    # Handle the return value (could be image or tuple)
                    if isinstance(result, tuple):
                        image, pipeline_generation_time = result
                        generation_time = pipeline_generation_time
                    else:
                        image = result
                        generation_time = time.time() - start_time
                else:
                    pipeline_result = pipeline.pipeline(
                        prompt=args.prompt,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        height=args.height,
                        width=args.width
                    )
                    image = pipeline_result.images[0] if hasattr(pipeline_result, 'images') else pipeline_result
                    generation_time = time.time() - start_time
                
                if hasattr(image, 'save'):
                    image.save(output_path)
                else:
                    from PIL import Image
                    if not isinstance(image, Image.Image):
                        image = Image.fromarray(image)
                    image.save(output_path)
                
                metadata = {"generation_time": generation_time}
            
            print(f"✅ Single image generated in {generation_time:.2f}s")
            print(f"📁 Saved to: {output_path}")
            
            # Save metadata for single image
            metadata_file = os.path.join(output_dir, "generation_metadata.json")
            full_metadata = {
                "prompt": args.prompt,
                "output_path": output_path,
                "generation_time": generation_time,
                "optimization_settings": {
                    "flash_attention": args.flash_attn,
                    "kv_cache": args.kv_cache,
                    "precision": args.precision,
                    "pruning": args.pruning
                },
                "generation_settings": {
                    "steps": args.steps,
                    "guidance_scale": args.guidance,
                    "height": args.height,
                    "width": args.width
                }
            }
            full_metadata.update(metadata)
            
            with open(metadata_file, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
        # Final cleanup
        print("\nTask completed. Performing final cleanup...")
        aggressive_memory_cleanup()
        print_memory_info()
        
    except Exception as e:
        print(f"\nError during image generation: {e}")
        aggressive_memory_cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()