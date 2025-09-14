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
import torch
import time
import gc
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
    parser.add_argument("--dataset", type=str, choices=["flickr8k", "coco"], default="none",
                        help="Dataset to use for generation (flickr8k or none for single prompt)")
    parser.add_argument("--num-images", type=int, default=10,
                        help="Number of images to generate from dataset")
    
    # Benchmark parameters
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmarks comparing Flash Attention vs standard attention")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials for benchmarking")
    
    return parser.parse_args()

def generate_dataset_images(pipeline, args):
    """Generate images from a dataset"""
    # Clean up memory before processing
    torch.cuda.empty_cache()
    gc.collect()
    
    # Process dataset
    try:
        if args.dataset == "flickr8k":
            print("Processing Flickr8k dataset...")
            flickr8k_dir = "flickr8k"
            images_dir = os.path.join(flickr8k_dir, "Images")
            captions_file = os.path.join(flickr8k_dir, "captions.txt")
            print("\n=== Loading Flickr8k Captions ===")
            # Process flickr8k dataset (now returns just captions dictionary)
            captions, image_dimensions = process_flickr8k(images_dir, captions_file)
            print(f"\n=== Generating Images for {len(captions)} Flickr8k Captions ===")
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
    except Exception as e:
        print(f"Error processing dataset: {e}")

    # Create output directory
    output_dir = os.path.join(
        "flash_attention",
        f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {args.num_images} images from {args.dataset} dataset")
    print(f"Flash Attention: {args.flash_attn}, KV Cache: {args.kv_cache}")
    print(f"Image size: {args.height}x{args.width}, Steps: {args.steps}")
    
    # Limit the number of images to generate
    if args.num_images < len(captions):
        # Get the top num_images captions (assume they are the first ones in the dictionary)
        image_keys = list(captions.keys())[:args.num_images]
        filtered_captions = {k: captions[k] for k in image_keys}
        print(f"Selected the top {args.num_images} captions from the dataset")
    else:
        filtered_captions = captions
        print(f"Using all {len(captions)} captions from the dataset")
        
    try:
        # Generate images with memory-efficient settings
        pipeline.generate_images_with_dataset(
            image_filename_to_caption=filtered_captions,
            output_dir=output_dir,
            num_images=args.num_images,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            use_cache=args.kv_cache,
            use_flash=args.flash_attn,
            height=args.height,
            width=args.width
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory during dataset generation: {e}")
        
        # Clean up memory
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
            
            # Generate image
            _, generation_time, peak_memory = generate_image_and_monitor(
                pipeline=pipeline,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed if args.seed else trial + 1000,
                use_cache=config["kv_cache"],
                use_flash=config["flash"],
                height=args.height,
                width=args.width
            )
            
            times.append(generation_time)
            memories.append(peak_memory)
            
            # Add a small delay between trials
            time.sleep(1)
        
        # Compute statistics
        avg_time = sum(times) / len(times)
        avg_memory = sum(memories) / len(memories)
        
        results[config["name"]] = {
            "avg_time": avg_time,
            "avg_memory": avg_memory,
            "times": times,
            "memories": memories,
        }
        
        print(f"  Average time: {avg_time:.2f} seconds")
        print(f"  Average memory: {avg_memory:.2f} MB")
    
    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    # Find baseline for comparison
    baseline_time = results["Standard Attention Only"]["avg_time"]
    baseline_memory = results["Standard Attention Only"]["avg_memory"]
    
    for name, data in results.items():
        time_speedup = baseline_time / data["avg_time"]
        memory_change = (data["avg_memory"] - baseline_memory) / baseline_memory * 100
        
        print(f"\n{name}:")
        print(f"  Time: {data['avg_time']:.2f} seconds ({time_speedup:.2f}x speedup)")
        print(f"  Memory: {data['avg_memory']:.2f} MB ({memory_change:+.2f}% change)")
    
    # Save benchmark results
    benchmark_dir = os.path.join(args.output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    import json
    with open(os.path.join(benchmark_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

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
    print(f"Precision: {args.precision}")
    print(f"Pruning: {args.pruning}")
    print(f"KV Cache: {args.kv_cache}")
    print(f"Flash Attention: {args.flash_attn}")
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
        print(f"\nGenerating images using {args.dataset} dataset...")
        generate_dataset_images(pipeline, args)
            
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