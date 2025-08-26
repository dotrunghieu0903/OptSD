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
from pycocotools.coco import COCO
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# Memory utilities for optimization
def setup_memory_optimizations():
    """Apply memory optimizations to avoid CUDA OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class KVCacheAttention(torch.nn.Module):
    """Attention module with Key-Value caching for efficient image generation"""
    
    def __init__(self, attn_module):
        super().__init__()
        # Copy attributes from original module
        self.original_attn = attn_module
        
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
        
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        """Forward pass with KV caching following the step-by-step process:
        1. First Generation: Calculate and store initial KV values
        2. Next Steps: Retrieve stored KV values and add new ones
        3. Efficient Attention: Calculate attention using cached K and V with new Q
        4. Update Input: Add newly generated features and continue
        """
        # If caching is not enabled or no cache key provided, use original forward
        if not self.cache_enabled or not kwargs.get('use_cache', False):
            return self.original_attn(hidden_states, encoder_hidden_states, attention_mask, **kwargs)

        timestep = kwargs.get('timestep', 0)
        cache_key = f"step_{timestep}"
        is_first_generation = cache_key not in self.kv_cache

        if is_first_generation:
            # Step 1: First Generation - Calculate and store initial KV values
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            query = self.to_q(hidden_states)
            
            # Store KV values in cache
            self.kv_cache[cache_key] = (key, value)
            
        else:
            # Step 2: Next Steps - Retrieve stored KV values
            cached_key, cached_value = self.kv_cache[cache_key]
            
            # Calculate new KV values for the current step
            new_key = self.to_k(hidden_states)
            new_value = self.to_v(hidden_states)
            
            # Concatenate with cached values
            key = torch.cat([cached_key, new_key], dim=1)
            value = torch.cat([cached_value, new_value], dim=1)
            
            # Update cache with new concatenated values
            self.kv_cache[cache_key] = (key, value)
            
            # Calculate new query
            query = self.to_q(hidden_states)
        
        # Step 3: Efficient Attention Computation
        attention_output = self._compute_attention(query, key, value, attention_mask)
        
        # Step 4: Input is automatically updated for next step
        return attention_output

    def _compute_attention(self, query, key, value, attention_mask=None):
        """Compute attention using provided query, key, value tensors"""
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        
        # Apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Apply softmax
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Calculate attention output
        hidden_states = torch.matmul(attention_probs, value)
        
        return hidden_states


class OptimizedCocoGenerator:
    """Image generator with KV caching optimization for COCO dataset"""
    
    def __init__(self, model_path="stabilityai/stable-diffusion-2-1"):
        """Initialize the generator with KV caching"""
        self.model_path = model_path
        self.pipeline = None
        self.attention_modules = []
        self.load_model()
        
    def load_model(self):
        """Load model and apply KV caching to transformer blocks"""
        print(f"Loading model from {self.model_path}...")
        
        # Apply memory optimizations
        setup_memory_optimizations()
        
        # Load the model
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        # Move to CUDA if available
        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to("cuda")
            
        # Apply KV caching to transformer blocks
        self._apply_kv_caching()
        
        print("Model loaded and KV caching applied")
        
    def _apply_kv_caching(self):
        """Apply KV caching to attention modules in the model"""
        # Find all attention modules in the UNet
        if hasattr(self.pipeline, "unet"):
            for name, module in self.pipeline.unet.named_modules():
                if any(attn_type in module.__class__.__name__.lower() for attn_type in ["attention", "attn"]):
                    # Apply KV caching to this attention module
                    kv_cached_attn = KVCacheAttention(module)
                    kv_cached_attn.enable_caching()
                    
                    # Replace the original attention module
                    parent_path = name.rsplit(".", 1)[0] if "." in name else ""
                    child_name = name.rsplit(".", 1)[1] if "." in name else name
                    
                    if parent_path:
                        parent = self.pipeline.unet.get_submodule(parent_path)
                        setattr(parent, child_name, kv_cached_attn)
                    else:
                        setattr(self.pipeline.unet, child_name, kv_cached_attn)
                        
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
    
    def generate(self, prompt, num_inference_steps=50, use_cache=True):
        """Generate image from prompt using KV caching"""
        # Manage cache state
        if not use_cache:
            self.clear_cache()
            self.disable_caching()
        else:
            self.enable_caching()
            self.clear_cache()  # Clear for new generation
        
        # Generate image
        image = self.pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            use_cache=use_cache
        ).images[0]
        
        return image
    
    def benchmark(self, prompts, num_inference_steps=50, num_runs=5, use_cache=True):
        """Benchmark image generation with and without KV caching"""
        results = {
            "with_cache": [],
            "without_cache": []
        }
        
        # Run with KV caching
        print(f"Benchmarking with KV caching {'enabled' if use_cache else 'disabled'}...")
        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")
            
            for i, prompt in enumerate(prompts):
                # Time the generation
                start_time = time.time()
                
                # Generate image
                _ = self.generate(prompt, num_inference_steps=num_inference_steps, use_cache=use_cache)
                
                generation_time = time.time() - start_time
                results["with_cache" if use_cache else "without_cache"].append(generation_time)
                
                print(f"  Prompt {i+1}: {generation_time:.4f} seconds")
                
                # Force memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
        # If we just ran with caching, run without it
        if use_cache:
            print("Benchmarking with KV caching disabled...")
            for run in range(num_runs):
                print(f"Run {run+1}/{num_runs}")
                
                for i, prompt in enumerate(prompts):
                    # Time the generation
                    start_time = time.time()
                    
                    # Generate image
                    _ = self.generate(prompt, num_inference_steps=num_inference_steps, use_cache=False)
                    
                    generation_time = time.time() - start_time
                    results["without_cache"].append(generation_time)
                    
                    print(f"  Prompt {i+1}: {generation_time:.4f} seconds")
                    
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


def load_coco_captions(annotation_file):
    """Load captions from COCO dataset"""
    try:
        coco = COCO(annotation_file)
        return coco
    except Exception as e:
        print(f"Error loading COCO annotations: {e}")
        return None


def main():
    """Main function for running image generation with KV caching"""
    parser = argparse.ArgumentParser(description="Generate images from COCO captions with KV caching")
    parser.add_argument("--annotation_file", type=str, required=True,
                        help="Path to COCO annotation file")
    parser.add_argument("--output_dir", type=str, default="generated_images",
                        help="Directory to save generated images")
    parser.add_argument("--num_images", type=int, default=5,
                        help="Number of images to generate")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark comparing cached vs. non-cached performance")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs for benchmarking")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the model")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load COCO dataset
    coco = load_coco_captions(args.annotation_file)
    if coco is None:
        return
    
    # Get random image annotations
    ann_ids = coco.getAnnIds()
    selected_ann_ids = np.random.choice(ann_ids, args.num_images, replace=False)
    annotations = coco.loadAnns(selected_ann_ids)
    
    # Create optimized generator
    generator = OptimizedCocoGenerator(model_path=args.model_path)
    
    if args.benchmark:
        # Get captions for benchmarking
        prompts = [ann['caption'] for ann in annotations]
        
        # Run benchmark
        generator.benchmark(prompts, num_inference_steps=args.num_steps, num_runs=args.num_runs)
    else:
        # Generate images from captions
        for i, ann in enumerate(annotations):
            caption = ann['caption']
            print(f"\nGenerating image {i+1}/{args.num_images}")
            print(f"Caption: {caption}")
            
            # Generate image
            image = generator.generate(caption, num_inference_steps=args.num_steps)
            
            # Save image
            output_path = os.path.join(args.output_dir, f"generated_{i+1}.png")
            image.save(output_path)
            print(f"Image saved as {output_path}")


if __name__ == "__main__":
    main()
