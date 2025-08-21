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
        # If caching is not enabled or we don't have a cache_id, use original forward
        if not self.cache_enabled or cache_id is None or not use_cache:
            return self.original_attn(q, k, v, mask=mask)
        
        # Check if we have cached values for this input
        if cache_id in self.kv_cache:
            cached_k, cached_v = self.kv_cache[cache_id]
            
            # Use cached k, v with new q
            attention_output = self._compute_attention(q, cached_k, cached_v, mask)
            return attention_output
        else:
            # Compute original attention
            attention_output = self.original_attn(q, k, v, mask=mask)
            
            # Cache k, v for future use
            self.kv_cache[cache_id] = (k, v)
            
            return attention_output
            
    def _compute_attention(self, q, k, v, attn_mask=None):
        """Compute attention using provided query, key, value tensors"""
        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
            
        # Apply softmax
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # Calculate attention output
        attn_output = torch.matmul(attn_probs, v)
        
        return attn_output


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
        
        # Load the SAM model
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
            print(f"Error loading SAM model: {e}")
            raise
            
        # Move to CUDA if available
        if torch.cuda.is_available():
            self.sam = self.sam.to("cuda")
            
        # Create predictor
        self.predictor = SamPredictor(self.sam)
        
        # Apply KV caching to transformer blocks
        self._apply_kv_caching()
        
        print("SAM model loaded and KV caching applied")
        
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
        """Set the image for processing"""
        self.predictor.set_image(image)
        
        # Generate image embedding with caching
        self.enable_caching()
        # Clear cache first to ensure clean state
        self.clear_cache()
        # Set unique cache ID for this image
        for i, module in enumerate(self.attention_modules):
            cache_id = f"image_{hash(str(image.sum()))}_module_{i}"
            module.cache_id = cache_id
    
    def predict(self, prompts, use_cache=True):
        """Run prediction with the current image and given prompts"""
        # If caching is disabled, clear cache to ensure no reuse
        if not use_cache:
            self.clear_cache()
            self.disable_caching()
        else:
            self.enable_caching()
            
        # Make predictions
        masks, scores, logits = self.predictor.predict(
            point_coords=prompts.get('point_coords', None),
            point_labels=prompts.get('point_labels', None),
            box=prompts.get('box', None),
            mask_input=prompts.get('mask_input', None),
            multimask_output=prompts.get('multimask_output', True),
        )
        
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


def main():
    """Main function for running the SAM model with KV caching"""
    parser = argparse.ArgumentParser(description="Run SAM 2.1 with KV caching optimization")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
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
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Get model path from config if not provided in args
    if args.model_path is None and config is not None:
        model_name = config["models"]["SAM 2.1"]["path"]
        args.model_path = model_name
    
    # Load the image
    try:
        image = cv2.imread(args.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Create optimized model
    sam = OptimizedSamModel(model_path=args.model_path)
    
    # Set the image
    sam.set_image(image)
    
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
