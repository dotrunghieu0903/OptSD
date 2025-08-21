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
from diffusers import DiffusionPipeline
from diffusers.models import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# Memory utilities for optimization
def setup_memory_optimizations():
    """Apply memory optimizations to avoid CUDA OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class KVCacheUNet(UNet2DConditionModel):
    """UNet model with Key-Value caching for transformer blocks"""
    
    def __init__(self, unet):
        # Initialize with the same parameters as the original UNet
        super().__init__(
            sample_size=unet.config.sample_size,
            in_channels=unet.config.in_channels,
            out_channels=unet.config.out_channels,
            down_block_types=unet.config.down_block_types,
            up_block_types=unet.config.up_block_types,
            block_out_channels=unet.config.block_out_channels,
            attention_head_dim=unet.config.attention_head_dim,
            cross_attention_dim=unet.config.cross_attention_dim,
            use_linear_projection=unet.config.get("use_linear_projection", False),
            layers_per_block=unet.config.layers_per_block,
        )
        
        # Copy weights from the original UNet
        self.load_state_dict(unet.state_dict())
        
        # Initialize KV cache storage
        self.kv_cache = {}
        
    def _apply_kv_caching_to_attention_processor(self):
        """Apply KV caching to the attention processors in the model"""
        for name, module in self.named_modules():
            if hasattr(module, "processor"):
                if "attn1" in name:  # Self-attention
                    # Apply self-attention KV caching
                    original_forward = module.processor.forward
                    module.processor.name = name
                    module.processor.original_forward = original_forward
                    module.processor.forward = self._make_kv_cached_attn_forward(module.processor)
                    
                elif "attn2" in name:  # Cross-attention
                    # Apply cross-attention KV caching
                    original_forward = module.processor.forward
                    module.processor.name = name
                    module.processor.original_forward = original_forward
                    module.processor.forward = self._make_kv_cached_cross_attn_forward(module.processor)
    
    def _make_kv_cached_attn_forward(self, processor):
        """Create a cached forward function for self-attention"""
        def cached_forward(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
            cache_key = f"{processor.name}_{kwargs.get('timestep', 0)}"
            
            if cache_key in self.kv_cache and kwargs.get('use_cache', False):
                # Use cached KV
                key, value = self.kv_cache[cache_key]
                
                # We still need to compute query
                query = attn.to_q(hidden_states)
                
                # Process query with cached key, value
                attn_output = self._compute_attention(
                    attn, query, key, value, attention_mask, kwargs.get('scale', 1.0)
                )
                
                return attn_output
            else:
                # Compute original forward pass
                output = processor.original_forward(
                    attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kwargs
                )
                
                # Cache KV if requested
                if kwargs.get('use_cache', False):
                    # Cache key and value
                    key = attn.to_k(hidden_states)
                    value = attn.to_v(hidden_states)
                    self.kv_cache[cache_key] = (key, value)
                
                return output
        
        return cached_forward
    
    def _make_kv_cached_cross_attn_forward(self, processor):
        """Create a cached forward function for cross-attention"""
        def cached_forward(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
            # For cross attention, we cache based on the combination of timestep and text embedding
            # Use a hash of the encoder_hidden_states to identify the text embedding
            enc_hash = str(hash(str(encoder_hidden_states.sum().item())))
            cache_key = f"{processor.name}_{kwargs.get('timestep', 0)}_{enc_hash}"
            
            if cache_key in self.kv_cache and kwargs.get('use_cache', False):
                # Use cached KV
                key, value = self.kv_cache[cache_key]
                
                # We still need to compute query
                query = attn.to_q(hidden_states)
                
                # Process query with cached key, value
                attn_output = self._compute_attention(
                    attn, query, key, value, attention_mask, kwargs.get('scale', 1.0)
                )
                
                return attn_output
            else:
                # Compute original forward pass
                output = processor.original_forward(
                    attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kwargs
                )
                
                # Cache KV if requested
                if kwargs.get('use_cache', False) and encoder_hidden_states is not None:
                    # Cache key and value
                    key = attn.to_k(encoder_hidden_states)
                    value = attn.to_v(encoder_hidden_states)
                    self.kv_cache[cache_key] = (key, value)
                
                return output
        
        return cached_forward
    
    def _compute_attention(self, attn, query, key, value, attention_mask=None, scale=1.0):
        """Compute attention using provided query, key, value tensors"""
        # Prepare inputs
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Get attention output
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states
    
    def clear_cache(self):
        """Clear KV cache"""
        self.kv_cache = {}
    
    def enable_kv_caching(self):
        """Enable KV caching in the model"""
        self._apply_kv_caching_to_attention_processor()
        print("KV caching enabled for UNet")
    
    def forward(self, *args, **kwargs):
        """Forward method with KV caching support"""
        # Extract caching parameters
        use_cache = kwargs.pop("use_cache", False)
        
        # Run the standard forward pass
        return super().forward(*args, **kwargs)


class OptimizedSANAPipeline:
    """Pipeline for the SANA model with KV caching optimization"""
    
    def __init__(self, model_path="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"):
        """Initialize the pipeline with KV caching"""
        self.model_path = model_path
        self.pipeline = None
        self.kv_cached_unet = None
        self.load_model()
        
    def load_model(self):
        """Load the SANA model and apply KV caching"""
        print(f"Loading model from {self.model_path}...")
        
        # Apply memory optimizations
        setup_memory_optimizations()
        
        # Load the pipeline
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        
        # Move to GPU
        self.pipeline = self.pipeline.to("cuda")
        
        # Extract the UNet model
        unet = self.pipeline.unet
        
        # Create KV cached UNet
        self.kv_cached_unet = KVCacheUNet(unet)
        self.kv_cached_unet.to(device=unet.device, dtype=unet.dtype)
        
        # Enable KV caching
        self.kv_cached_unet.enable_kv_caching()
        
        # Replace the UNet in the pipeline
        self.pipeline.unet = self.kv_cached_unet
        
        print(f"Model loaded and KV caching enabled")
    
    def generate_image(self, prompt, negative_prompt="", num_inference_steps=30, 
                      guidance_scale=7.5, seed=None, use_cache=True, height=1024, width=1024):
        """Generate an image using the KV cached pipeline"""
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Clear KV cache before starting new generation
        if use_cache:
            self.kv_cached_unet.clear_cache()
        
        # Generate the image
        start_time = time.time()
        
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            use_cache=use_cache,  # This flag will be passed to the UNet forward method
        ).images[0]
        
        generation_time = time.time() - start_time
        
        print(f"Image generated in {generation_time:.2f} seconds with KV caching {'enabled' if use_cache else 'disabled'}")
        
        return image, generation_time
    
    def benchmark(self, prompt, negative_prompt="", num_inference_steps=30, guidance_scale=7.5, 
                 num_runs=5, use_cache=True, height=1024, width=1024):
        """Benchmark image generation with and without KV caching"""
        results = {
            "with_cache": [],
            "without_cache": []
        }
        
        # Run with KV caching
        print(f"Benchmarking with KV caching {'enabled' if use_cache else 'disabled'}...")
        for i in range(num_runs):
            _, generation_time = self.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=i,  # Use run index as seed for reproducibility
                use_cache=use_cache,
                height=height,
                width=width
            )
            results["with_cache" if use_cache else "without_cache"].append(generation_time)
            
            # Force memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
        
        # Run without KV caching (if we just ran with it)
        if use_cache:
            print("Benchmarking with KV caching disabled...")
            for i in range(num_runs):
                _, generation_time = self.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=i,  # Use run index as seed for reproducibility
                    use_cache=False,
                    height=height,
                    width=width
                )
                results["without_cache"].append(generation_time)
                
                # Force memory cleanup
                torch.cuda.empty_cache()
                gc.collect()
        
        # Calculate statistics
        with_cache_avg = sum(results["with_cache"]) / len(results["with_cache"]) if results["with_cache"] else 0
        without_cache_avg = sum(results["without_cache"]) / len(results["without_cache"]) if results["without_cache"] else 0
        
        speedup = without_cache_avg / with_cache_avg if with_cache_avg > 0 else 0
        
        print(f"\n===== Benchmark Results =====")
        print(f"Average time with KV caching: {with_cache_avg:.2f} seconds")
        if without_cache_avg > 0:
            print(f"Average time without KV caching: {without_cache_avg:.2f} seconds")
            print(f"Speedup factor: {speedup:.2f}x")
        
        return results
    
    def interactive_session(self, initial_prompt, num_inference_steps=30, guidance_scale=7.5,
                          negative_prompt="", height=1024, width=1024):
        """Run an interactive session with the same image and different prompts"""
        print("Starting interactive session...")
        print("Generating initial image...")
        
        # Generate initial image with KV cache enabled
        with torch.no_grad():
            image, past_generation_time = self.generate_image(
                prompt=initial_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                use_cache=True,
                height=height,
                width=width
            )
        
        # Display or save the initial image
        image_filename = f"initial_image_{int(time.time())}.png"
        image.save(image_filename)
        print(f"Initial image saved as {image_filename}")
        
        # Interactive loop
        while True:
            updated_prompt = input("\nEnter a new prompt (or 'quit' to exit): ")
            if updated_prompt.lower() == 'quit':
                break
            
            print(f"Generating image for: {updated_prompt}")
            
            # Generate new image with the same cached KV values
            with torch.no_grad():
                # Don't clear the cache so we can reuse it
                new_image, generation_time = self.generate_image(
                    prompt=updated_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    use_cache=True,  # Use cache
                    height=height,
                    width=width
                )
            
            # Save the new image
            new_image_filename = f"image_{int(time.time())}.png"
            new_image.save(new_image_filename)
            print(f"New image saved as {new_image_filename}")
            print(f"Generation time: {generation_time:.2f} seconds (vs. initial: {past_generation_time:.2f} seconds)")
            
            # Update for comparison with next generation
            past_generation_time = generation_time


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
    """Main function for running the SANA model with KV caching"""
    parser = argparse.ArgumentParser(description="Run SANA model with KV caching optimization")
    parser.add_argument("--prompt", type=str, default="a photo of a dog on a beach",
                        help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt for image generation")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for image generation")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark comparing cached vs. non-cached performance")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs for benchmarking")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode to test multiple prompts")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the SANA model")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Get model path from config if not provided in args
    if args.model_path is None and config is not None:
        model_name = config["models"]["SANA 1.6B"]["path"]
        args.model_path = model_name
    
    # Create optimized pipeline
    pipeline = OptimizedSANAPipeline(model_path=args.model_path)
    
    if args.benchmark:
        # Run benchmark
        pipeline.benchmark(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            num_runs=args.num_runs,
            height=args.height,
            width=args.width
        )
    elif args.interactive:
        # Run interactive session
        pipeline.interactive_session(
            initial_prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width
        )
    else:
        # Generate a single image
        image, _ = pipeline.generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            height=args.height,
            width=args.width
        )
        
        # Save the image
        output_filename = f"output_{int(time.time())}.png"
        image.save(output_filename)
        print(f"Image saved as {output_filename}")


if __name__ == "__main__":
    main()
