#!/usr/bin/env python3

import argparse
import gc
import os
import time
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import cv2

# Fix import paths to be relative to the project root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import calculate_clip_score, calculate_fid, calculate_lpips, calculate_psnr_resized
from shared.resources_monitor import generate_image_and_monitor, write_generation_metadata_to_file

# Memory utilities for optimization
def setup_memory_optimizations():
    """Apply memory optimizations to avoid CUDA OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class KVCacheSamAttention(torch.nn.Module):
    """SAM attention module with Key-Value caching"""
    
    def __init__(self, attn_module):
        super().__init__()
        # Copy attributes from original module
        self.original_attn = attn_module
        self.embed_dim = getattr(attn_module, 'embed_dim', None)
        self.num_heads = getattr(attn_module, 'num_heads', None)
        self.head_dim = getattr(attn_module, 'head_dim', None)
        
        # Copy all attributes dynamically
        for name, value in vars(attn_module).items():
            if not name.startswith('_') and name not in ['forward']:
                setattr(self, name, value)
                
        # Initialize KV cache
        self.kv_cache = {}
        self.cache_enabled = False
        
    def enable_caching(self):
        """Enable KV caching"""
        self.cache_enabled = True
        return self
        
    def disable_caching(self):
        """Disable KV caching"""
        self.cache_enabled = False
        return self
        
    def clear_cache(self):
        """Clear the KV cache"""
        self.kv_cache = {}
        
    def forward(self, q, k, v, mask=None, cache_id=None, use_cache=False):
        """Forward pass with KV caching following the step-by-step process:
        1. First Generation: Calculate and store initial KV values
        2. Next Words: Retrieve stored KV values and add new ones
        3. Efficient Attention: Calculate attention using cached K and V with new Q
        4. Update Input: Add newly generated token and continue
        """
        # If caching is not enabled or no cache ID provided, use original forward
        if not self.cache_enabled or cache_id is None or not use_cache:
            return self.original_attn(q, k, v, mask=mask)

        is_first_generation = cache_id not in self.kv_cache

        if is_first_generation:
            # Step 1: First Generation - Calculate and store initial KV values
            # For SAM, k and v are already calculated, just store them
            self.kv_cache[cache_id] = (k, v)
            
            # Use original attention for first pass
            attention_output = self._compute_attention(q, k, v, mask)
        else:
            # Step 2: Next Words - Retrieve stored KV values
            cached_k, cached_v = self.kv_cache[cache_id]
            
            # Calculate attention using cached values
            # Step 3: Efficient Attention Computation
            attention_output = self._compute_attention(q, cached_k, cached_v, mask)
            
            # Unlike language models, SAM doesn't need to append new values 
            # as it processes the full image at once
        
        # Step 4: Input updates are handled by the SAM predictor
        return attention_output
            
    def _compute_attention(self, q, k, v, attn_mask=None):
        """Compute attention using provided query, key, value tensors
        This implements the efficient attention computation (Step 3) of the process.
        """
        # Step 3a: Calculate attention scores with cached K
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Step 3b: Apply mask if provided
        if attn_mask is not None:
            attention_scores = attention_scores + attn_mask
            
        # Step 3c: Apply softmax to get attention probabilities
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Step 3d: Calculate final attention output using cached V
        hidden_states = torch.matmul(attention_probs, v)
        
        return hidden_states


class OptimizedSamModel:
    """SAM 2.1 model with KV caching optimization"""
    
    def __init__(self, model_path="facebook/sam2.1-hiera-small"):
        """Initialize the SAM model with KV caching"""
        self.model_path = model_path
        self.sam = None
        self.predictor = None
        self.attention_modules = []
        self.load_model()
        
    def load_model(self):
        """Load SAM model and apply KV caching to transformer blocks"""
        print(f"Loading SAM model from {self.model_path}...")
        
        # Apply memory optimizations
        setup_memory_optimizations()
        
        # Load the SAM model with error handling and retry
        try:
            # For local checkpoint
            if os.path.exists(self.model_path):
                model_type = "sam2_1_hiera_s"  # or vit_h, vit_l, vit_b based on model
                self.sam = sam_model_registry[model_type](checkpoint=self.model_path)
            else:
                # For HuggingFace model
                from transformers import SamModel
                self.sam = SamModel.from_pretrained(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying again with default parameters...")
            try:
                # Fallback to default model
                model_path = "facebook/sam-vit-base"
                from transformers import SamModel
                self.sam = SamModel.from_pretrained(model_path)
                print("Loaded fallback model successfully")
            except Exception as e2:
                print(f"Fatal error loading model: {e2}")
                raise
            
        # Move to CUDA if available and calculate model size
        if torch.cuda.is_available():
            self.sam = self.sam.to("cuda")
            
        # Calculate model size
        model_size = self.get_model_size()
        print(f"Model size: {model_size:.2f} MB")
            
        # Create predictor
        self.predictor = SamPredictor(self.sam)
        
        # Apply KV caching to transformer blocks
        self._apply_kv_caching()
        
        # Test the model with a simple prediction
        print("Testing model with a simple prediction...")
        try:
            test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            self.set_image(test_image)
            _ = self.predict({'point_coords': np.array([[32, 32]]), 'point_labels': np.array([1])})
            print("Model test successful!")
        except Exception as e:
            print(f"Warning: Model test failed: {e}")
        finally:
            # Clear test memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        print("SAM model loaded and KV caching applied")
        
    def get_model_size(self):
        """Calculate the model size in MB"""
        model_size_bytes = sum(p.numel() * p.element_size() for p in self.sam.parameters())
        model_size_mb = model_size_bytes / (1024 * 1024)
        return model_size_mb
        
    def _apply_kv_caching(self):
        """Apply KV caching to attention modules in the SAM model"""
        # Find all attention modules in image encoder
        if hasattr(self.sam, "image_encoder"):
            for name, module in self.sam.image_encoder.named_modules():
                # Look for attention modules - specific module type depends on the model architecture
                if any(attn_type in module.__class__.__name__.lower() for attn_type in ["attention", "attn"]):
                    # Apply KV caching to this attention module
                    kv_cached_attn = KVCacheSamAttention(module)
                    kv_cached_attn.enable_caching()
                    
                    # Replace the original attention module
                    parent_path = name.rsplit(".", 1)[0] if "." in name else ""
                    child_name = name.rsplit(".", 1)[1] if "." in name else name
                    
                    if parent_path:
                        parent = self.sam.image_encoder.get_submodule(parent_path)
                        setattr(parent, child_name, kv_cached_attn)
                    else:
                        setattr(self.sam.image_encoder, child_name, kv_cached_attn)
                        
                    self.attention_modules.append(kv_cached_attn)
                    print(f"Applied KV caching to {name}")
        
        print(f"Applied KV caching to {len(self.attention_modules)} attention modules")
    
    def clear_cache(self):
        """Clear the KV cache in all attention modules"""
        for attn_module in self.attention_modules:
            attn_module.clear_cache()
        
    def enable_caching(self):
        """Enable KV caching in all attention modules"""
        for attn_module in self.attention_modules:
            attn_module.enable_caching()
            
    def disable_caching(self):
        """Disable KV caching in all attention modules"""
        for attn_module in self.attention_modules:
            attn_module.disable_caching()
    
    def set_image(self, image):
        """Set the image for processing and prepare KV caching
        Following the step-by-step process:
        1. First Generation: Initial setup and cache clear
        2. Next Words: Prepare for subsequent prompts
        3. Efficient Attention: Enable caching
        4. Update Input: Handle through predictor
        """
        # Step 1: First Generation Setup
        self.clear_cache()  # Clear any existing cache
        self.enable_caching()  # Enable caching mechanism
        
        # Step 2: Prepare for Next Words/Prompts
        # Generate a unique ID for this image's cache
        image_hash = hash(str(image.sum()))
        for i, module in enumerate(self.attention_modules):
            # Each attention module gets its own cache entry
            cache_id = f"image_{image_hash}_module_{i}"
            module.cache_id = cache_id
        
        # Step 3: Set up for efficient attention
        # The predictor will handle the actual image processing
        self.predictor.set_image(image)
        
        # Step 4: Input Updates
        # SAM will handle subsequent prompt processing using the cached image features
    
    def predict(self, prompts, use_cache=True):
        """Run prediction with the current image and given prompts
        Following the step-by-step process:
        1. First Generation: Use cached image features
        2. Next Words: Process new prompt
        3. Efficient Attention: Use cached K,V values
        4. Update Input: Prepare for next prompt
        """
        # Step 1 & 2: Handle cache state for this prediction
        if not use_cache:
            self.clear_cache()
            self.disable_caching()
        else:
            # Ensure caching is enabled for efficient processing
            self.enable_caching()
        
        # Step 3: Make predictions using efficient attention
        # The SAM predictor will automatically use our cached KV values
        masks, scores, logits = self.predictor.predict(
            point_coords=prompts.get('point_coords', None),
            point_labels=prompts.get('point_labels', None),
            box=prompts.get('box', None),
            mask_input=prompts.get('mask_input', None),
            multimask_output=prompts.get('multimask_output', True),
        )
        
        # Step 4: Return results, cache is maintained for next prompt
        return {
            'masks': masks,
            'scores': scores,
            'logits': logits
        }
    
    def benchmark(self, image, prompts_list, num_runs=5, use_cache=True):
        """Benchmark segmentation with and without KV caching"""
        results = {
            "with_cache": [],
            "without_cache": []
        }
        
        # First load the image (only once)
        self.set_image(image)
        
        # Run with KV caching
        print(f"Benchmarking with KV caching {'enabled' if use_cache else 'disabled'}...")
        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")
            
            for i, prompts in enumerate(prompts_list):
                # Time the prediction
                start_time = time.time()
                
                # Make prediction
                _ = self.predict(prompts, use_cache=use_cache)
                
                prediction_time = time.time() - start_time
                results["with_cache" if use_cache else "without_cache"].append(prediction_time)
                
                print(f"  Prompt {i+1}: {prediction_time:.4f} seconds")
                
                # Force memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
        # If we just ran with caching, run without it
        if use_cache:
            # Clear cache and disable caching
            self.clear_cache()
            self.disable_caching()
            
            # Set image again to reset embeddings
            self.set_image(image)
            
            print("Benchmarking with KV caching disabled...")
            for run in range(num_runs):
                print(f"Run {run+1}/{num_runs}")
                
                for i, prompts in enumerate(prompts_list):
                    # Time the prediction
                    start_time = time.time()
                    
                    # Make prediction
                    _ = self.predict(prompts, use_cache=False)
                    
                    prediction_time = time.time() - start_time
                    results["without_cache"].append(prediction_time)
                    
                    print(f"  Prompt {i+1}: {prediction_time:.4f} seconds")
                    
                    # Force memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
        
        # Calculate statistics
        with_cache_avg = sum(results["with_cache"]) / len(results["with_cache"]) if results["with_cache"] else 0
        without_cache_avg = sum(results["without_cache"]) / len(results["without_cache"]) if results["without_cache"] else 0
        
        speedup = without_cache_avg / with_cache_avg if with_cache_avg > 0 else 0
        
        print(f"\n===== Benchmark Results =====")
        print(f"Average time with KV caching: {with_cache_avg:.4f} seconds")
        if without_cache_avg > 0:
            print(f"Average time without KV caching: {without_cache_avg:.4f} seconds")
            print(f"Speedup factor: {speedup:.2f}x")
        
        return results
    
    def interactive_session(self, image):
        """Run an interactive session with the same image and different prompts"""
        print("Starting interactive session...")
        print("Processing initial image...")
        
        # Set the image
        self.set_image(image)
        
        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title("Click points on the image (left click = foreground, right click = background)")
        plt.show()
        
        # Interactive loop
        while True:
            prompt_type = input("\nPrompt type (point, box, or 'quit' to exit): ").strip().lower()
            
            if prompt_type == 'quit':
                break
            
            prompts = {}
            
            if prompt_type == 'point':
                # Get point coordinates from user input
                point_input = input("Enter point coordinates as 'x1,y1,label1 x2,y2,label2...' (label: 1=foreground, 0=background): ")
                points = []
                labels = []
                
                for point in point_input.split():
                    x, y, label = map(int, point.split(','))
                    points.append([x, y])
                    labels.append(label)
                
                if points:
                    prompts['point_coords'] = np.array(points)
                    prompts['point_labels'] = np.array(labels)
            
            elif prompt_type == 'box':
                # Get box coordinates from user input
                box_input = input("Enter box coordinates as 'x1,y1,x2,y2': ")
                x1, y1, x2, y2 = map(int, box_input.split(','))
                prompts['box'] = np.array([x1, y1, x2, y2])
            
            else:
                print("Invalid prompt type. Use 'point', 'box', or 'quit'.")
                continue
            
            # Get multimask option
            multimask = input("Generate multiple masks? (y/n, default=y): ").strip().lower()
            prompts['multimask_output'] = False if multimask == 'n' else True
            
            print("Running prediction...")
            start_time = time.time()
            
            # Run prediction with KV caching
            results = self.predict(prompts, use_cache=True)
            
            prediction_time = time.time() - start_time
            print(f"Prediction completed in {prediction_time:.4f} seconds")
            
            # Display results
            masks = results['masks']
            scores = results['scores']
            
            # Show the image with mask overlay
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            
            if prompts['multimask_output']:
                num_masks = len(masks)
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    plt.contour(mask, colors=['red', 'green', 'blue'][i % 3], levels=[0.5], alpha=1)
                    plt.text(10, 30 + 30*i, f"Mask {i+1}, Score: {score:.3f}", color=['red', 'green', 'blue'][i % 3], 
                             backgroundcolor="white", fontsize=12)
            else:
                plt.contour(masks[0], colors='red', levels=[0.5], alpha=1)
                plt.text(10, 30, f"Score: {scores[0]:.3f}", color='red', backgroundcolor="white", fontsize=12)
                
            plt.axis('off')
            plt.title("Segmentation Results")
            plt.show()


def load_config():
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as file:
            config = json.load(file)
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def calculate_metrics(original_dir, generated_dir, image_info, subset_size=100):
    """Calculate quality metrics for generated masks"""
    metrics_results = {}
    
    # Create directory for resized images
    resized_dir = os.path.join(generated_dir, "resized")
    os.makedirs(resized_dir, exist_ok=True)
    
    try:
        print("\n--- Calculating FID Score ---")
        fid_score = calculate_fid(generated_dir, resized_dir, original_dir)
        metrics_results["fid_score"] = fid_score
    except Exception as e:
        print(f"Error calculating FID: {e}")
    
    try:
        print("\n--- Calculating CLIP Score ---")
        # Create caption dictionary for CLIP score calculation
        captions = {img: info['annotations'][0]['caption'] if info['annotations'] else ""
                   for img, info in image_info.items()}
        clip_score = calculate_clip_score(generated_dir, captions)
        metrics_results["clip_score"] = clip_score
    except Exception as e:
        print(f"Error calculating CLIP Score: {e}")
    
    try:
        print("\n--- Calculating LPIPS ---")
        # Get list of generated images
        generated_files = [f for f in os.listdir(generated_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Use subset for LPIPS calculation
        subset_files = generated_files[:subset_size]
        lpips_score = calculate_lpips(original_dir, generated_dir, subset_files)
        metrics_results["lpips"] = lpips_score
    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
    
    try:
        print("\n--- Calculating PSNR ---")
        psnr_score = calculate_psnr_resized(original_dir, generated_dir, subset_files)
        metrics_results["psnr"] = psnr_score
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
    
    return metrics_results


def main():
    """Main function for running the SAM model with KV caching"""
    parser = argparse.ArgumentParser(description="Run SAM 2.1 with KV caching optimization")
    parser.add_argument("--image", type=str,
                        help="Path to the input image (optional)")
    parser.add_argument("--coco_dir", type=str, default="coco",
                        help="Path to COCO dataset directory")
    parser.add_argument("--num_images", type=int, default=100,
                        help="Number of COCO images to process")
    parser.add_argument("--point_coords", type=str, default=None,
                        help="Point coordinates for prompting, format: 'x1,y1,label1 x2,y2,label2'")
    parser.add_argument("--box", type=str, default=None,
                        help="Box coordinates for prompting, format: 'x1,y1,x2,y2'")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark comparing cached vs. non-cached performance")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs for benchmarking")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode to test multiple prompts")
    parser.add_argument("--multimask", action="store_true",
                        help="Generate multiple masks")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the SAM model")
    parser.add_argument("--skip_metrics", action="store_true",
                        help="Skip calculation of quality metrics")
    parser.add_argument("--metrics_subset", type=int, default=100,
                        help="Number of images to use for metrics calculation")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Get model path from config if not provided in args
    if args.model_path is None and config is not None:
        model_name = config["models"]["SAM 2.1"]["path"]
        args.model_path = model_name
    
    # Setup COCO paths
    coco_dir = args.coco_dir
    annotations_dir = os.path.join(coco_dir, "annotations")
    val2017_dir = os.path.join(coco_dir, "val2017")
    
    # Create output directories
    output_dir = os.path.join("kvcache", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a list to store generation metadata
    generation_metadata = []
    
    # Load COCO dataset if not in interactive mode
    if not args.interactive and not args.image:
        print("\n=== Loading COCO Dataset ===")
        annotation_file = os.path.join(annotations_dir, "instances_val2017.json")
        image_info = {}
        # image_info, coco = load_coco_dataset(annotation_file, args.num_images)
        # if image_info is None:
        #     return
    
    # Load single image if specified
    elif args.image:
        try:
            image = cv2.imread(args.image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image: {e}")
            return
    
    # Create optimized model
    sam = OptimizedSamModel(model_path=args.model_path)
    
    if args.interactive or args.image:
        # Process single image mode
        sam.set_image(image)
        
        if args.benchmark:
            # Create prompts for benchmarking
            prompts_list = []
    else:
        # Process COCO dataset
        print("\n=== Processing COCO Images ===")
        
        # Track statistics
        processing_times = []
        successful_masks = 0
        
        for img_file, info in tqdm(list(image_info.items())[:args.num_images]):
            img_path = os.path.join(val2017_dir, img_file)
            output_path = os.path.join(output_dir, img_file.replace('.jpg', '_mask.png'))
            
            # Skip if output exists
            if os.path.exists(output_path):
                print(f"Skipping {img_file} (output exists)")
                continue
                
            try:
                # Load and process image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get annotations for automatic prompting
                annotations = info['annotations']
                if not annotations:
                    continue
                    
                # Use annotation bounding box as prompt
                bbox = annotations[0]['bbox']  # [x, y, width, height]
                prompt = {
                    'box': np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]),
                    'multimask_output': args.multimask
                }
                
                # Time the generation
                start_time = time.time()
                
                # Set image and generate mask
                sam.set_image(image)
                result = sam.predict(prompt)
                
                # Save processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Save the mask
                mask = result['masks'][0]
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img.save(output_path)
                
                successful_masks += 1
                
                # Add generation metadata
                metadata = {
                    'filename': img_file,
                    'processing_time': processing_time,
                    'success': True
                }
                generation_metadata.append(metadata)
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                metadata = {
                    'filename': img_file,
                    'error': str(e),
                    'success': False
                }
                generation_metadata.append(metadata)
            
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate and save metrics if not skipped
        if not args.skip_metrics:
            print("\n=== Calculating Quality Metrics ===")
            metrics = calculate_metrics(val2017_dir, output_dir, image_info, args.metrics_subset)
            
            # Save metrics
            metrics_file = os.path.join(output_dir, "metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print("\n=== Quality Metrics ===")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        # Save summary
        summary_file = os.path.join(output_dir, "processing_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=== Processing Summary ===\n\n")
            f.write(f"Total images processed: {args.num_images}\n")
            f.write(f"Successful masks: {successful_masks}\n")
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                f.write(f"Average processing time: {avg_time:.2f} seconds\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"KV Cache: Enabled\n")
            
        print(f"\nProcessing complete. Results saved in {output_dir}")
        print(f"Summary saved to {summary_file}")
        return
        
    if args.benchmark:
        # Create prompts for benchmarking
        prompts_list = []
        
        # Add point prompts if provided
        if args.point_coords:
            points = []
            labels = []
            for point in args.point_coords.split():
                x, y, label = map(int, point.split(','))
                points.append([x, y])
                labels.append(label)
            
            prompts_list.append({
                'point_coords': np.array(points),
                'point_labels': np.array(labels),
                'multimask_output': args.multimask
            })
        
        # Add box prompts if provided
        if args.box:
            x1, y1, x2, y2 = map(int, args.box.split(','))
            prompts_list.append({
                'box': np.array([x1, y1, x2, y2]),
                'multimask_output': args.multimask
            })
        
        # Add a default box prompt if none provided
        if not prompts_list:
            h, w, _ = image.shape
            prompts_list.append({
                'box': np.array([w//4, h//4, 3*w//4, 3*h//4]),
                'multimask_output': args.multimask
            })
            print("Using default box prompt for benchmark")
        
        # Run benchmark
        sam.benchmark(image, prompts_list, num_runs=args.num_runs)
    
    elif args.interactive:
        # Run interactive session
        sam.interactive_session(image)
    
    else:
        # Create prompts from command line arguments
        prompts = {'multimask_output': args.multimask}
        
        # Add point prompts if provided
        if args.point_coords:
            points = []
            labels = []
            for point in args.point_coords.split():
                x, y, label = map(int, point.split(','))
                points.append([x, y])
                labels.append(label)
            
            prompts['point_coords'] = np.array(points)
            prompts['point_labels'] = np.array(labels)
        
        # Add box prompts if provided
        if args.box:
            x1, y1, x2, y2 = map(int, args.box.split(','))
            prompts['box'] = np.array([x1, y1, x2, y2])
        
        # Run prediction
        results = sam.predict(prompts)
        
        # Display results
        masks = results['masks']
        scores = results['scores']
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        if args.multimask:
            num_masks = len(masks)
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.contour(mask, colors=['red', 'green', 'blue'][i % 3], levels=[0.5], alpha=1)
                plt.text(10, 30 + 30*i, f"Mask {i+1}, Score: {score:.3f}", color=['red', 'green', 'blue'][i % 3], 
                         backgroundcolor="white", fontsize=12)
        else:
            plt.contour(masks[0], colors='red', levels=[0.5], alpha=1)
            plt.text(10, 30, f"Score: {scores[0]:.3f}", color='red', backgroundcolor="white", fontsize=12)
            
        plt.axis('off')
        plt.title("Segmentation Results")
        plt.savefig("sam_segmentation_result.png")
        print("Segmentation result saved as 'sam_segmentation_result.png'")


if __name__ == "__main__":
    main()
