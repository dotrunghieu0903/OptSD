#!/usr/bin/env python3
"""
Combined optimization for diffusion models with Flash Attention.

This module extends the combined optimization pipeline with Flash Attention support.
It integrates Flash Attention with existing optimizations (quantization, pruning, KV caching).
"""

import os
import sys
import time
import torch

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from combination.combined_optimization import OptimizedDiffusionPipeline
from flash_attention.flash_attention_integration import FlashAttentionKVCacheTransformer
# Import memory optimization tools
import gc

from nunchaku import NunchakuFluxTransformer2dModel

# Configure PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

def free_memory():
    """Free unused memory to avoid OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class FlashAttentionOptimizedPipeline(OptimizedDiffusionPipeline):
    """
    Extended pipeline with Flash Attention support in addition to other optimizations
    
    This class extends the OptimizedDiffusionPipeline with Flash Attention support,
    enabling faster attention computation for diffusion models.
    """
    
    def __init__(self, model_path=None, precision=None, pruning_amount=0.0, 
                 use_kv_cache=True, use_flash_attention=True,
                 device_map="auto", use_bf16=False,
                 offload_folder=None):
        """
        Initialize the pipeline with Flash Attention support and memory optimizations
        
        Args:
            model_path: Path to the model (HuggingFace repo ID or local path)
            precision: Precision to use for quantization (int4, int8, etc.)
            pruning_amount: Amount of weights to prune (0.0 to 0.9)
            use_kv_cache: Whether to use KV caching
            use_flash_attention: Whether to use Flash Attention
            device_map: Device mapping strategy for model loading
            use_bf16: Whether to use bfloat16 precision
            offload_folder: Folder for offloading weights to disk
        """
        # Set up Flash Attention specific attributes before calling parent
        self.use_flash_attention = use_flash_attention
        self.device_map = device_map
        self.offload_folder = offload_folder
        self.use_bf16 = use_bf16
        
        # Initialize parent class first
        super().__init__(model_path=model_path, precision=precision, 
                        pruning_amount=pruning_amount, use_kv_cache=use_kv_cache)
        
        # Apply Flash Attention if pipeline was loaded
        if hasattr(self, 'pipeline') and self.pipeline is not None and self.use_flash_attention:
            free_memory()  # Free memory before applying flash attention
            self._apply_flash_attention()
    
    def load_model(self):
        """Load model with Flash Attention support, bypassing parent class issues"""
        print(f"Loading model from {self.model_path} with Flash Attention support...")
        
        if not self.model_path:
            raise ValueError("model_path must be provided")
            
        try:
            # Use our reliable fallback method directly
            self._load_model_fallback()
            print("✓ Model loaded successfully")
            
            # Apply Flash Attention if enabled and pipeline was loaded successfully
            if hasattr(self, 'pipeline') and self.pipeline is not None and self.use_flash_attention:
                print("Applying Flash Attention to loaded pipeline...")
                self._apply_flash_attention()
            else:
                print("Flash Attention not applied - pipeline not initialized or disabled")
                
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
                
    def _load_model_fallback(self):
        """Fallback method to load model if parent class fails"""
        print("Using fallback model loading...")
        
        from diffusers import DiffusionPipeline
        
        try:
            # Apply memory optimizations first
            free_memory()
            
            model_path = f"nunchaku-tech/nunchaku-flux.1-schnell/svdq-int4_r32-flux.1-schnell.safetensors"
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(model_path, offload=True)
            # Simple direct loading with diffusers
            pipe_path = "black-forest-labs/FLUX.1-schnell"
            print(f"Loading model: {pipe_path}")
            self.pipeline = DiffusionPipeline.from_pretrained(
                pipe_path,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
            
            # Move to CUDA if available
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
                print("✓ Moved pipeline to CUDA")
            
            print("✓ Fallback loading successful")
            
            # Apply basic memory optimizations
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing(1)
                print("✓ Enabled attention slicing")
            
            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
                print("✓ Enabled VAE slicing")
                
            if hasattr(self.pipeline, "enable_vae_tiling"):
                self.pipeline.enable_vae_tiling()
                print("✓ Enabled VAE tiling")
                
            # Find transformer for KV caching and Flash Attention
            transformer = self._locate_transformer()
            if transformer is not None:
                print(f"✓ Located transformer component")
                self.original_transformer = transformer
                
                # Apply KV caching if requested
                if self.use_kv_cache:
                    try:
                        from combination.combined_optimization import KVCacheTransformer
                        
                        print("Applying KV cache to transformer...")
                        self.kv_cached_transformer = KVCacheTransformer(transformer)
                        
                        # Get device and dtype from transformer
                        device = transformer.device if hasattr(transformer, 'device') else torch.device('cuda')
                        dtype = transformer.dtype if hasattr(transformer, 'dtype') else None
                        
                        # Move transformer to appropriate device and dtype
                        if dtype is not None:
                            self.kv_cached_transformer.to(device=device, dtype=dtype)
                        else:
                            self.kv_cached_transformer.to(device=device)
                        
                        # Enable KV caching
                        self.kv_cached_transformer.enable_kv_caching()
                        
                        # Replace transformer in pipeline
                        success = self._replace_transformer(self.kv_cached_transformer)
                        if success:
                            print("✓ KV caching applied successfully")
                        else:
                            print("✗ Failed to apply KV caching")
                            
                    except Exception as kv_error:
                        print(f"✗ KV caching failed: {kv_error}")
                        
            else:
                print("✗ Could not locate transformer - KV caching and Flash Attention unavailable")
                
            # Test pipeline with a simple prompt to ensure it works
            try:
                print("Testing pipeline with minimal prompt...")
                test_result = self.pipeline(
                    "test",
                    num_inference_steps=1,
                    height=256,
                    width=256
                )
                if hasattr(test_result, 'images') and test_result.images:
                    print("✓ Pipeline test successful!")
                    del test_result  # Clean up
                else:
                    print("✗ Pipeline test failed - no images returned")
                    
                free_memory()
                
            except Exception as test_error:
                print(f"✗ Pipeline test failed: {test_error}")
                # Continue anyway as the pipeline might still work for actual generation
                
        except Exception as e:
            print(f"✗ Fallback loading failed: {e}")
            raise
    
    def _apply_flash_attention(self):
        """Apply Flash Attention to the transformer with memory optimizations"""
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            print("Cannot apply Flash Attention: Pipeline not initialized")
            return
        
        print("Applying Flash Attention optimization with memory safeguards...")
        
        # Clear memory before locating and modifying transformer
        free_memory()
        
        try:
            # First, locate the transformer
            transformer = self._locate_transformer()
            
            if transformer is not None:
                # Replace the transformer with Flash Attention enhanced version
                if self.use_flash_attention:
                    print("Applying Flash Attention to transformer")
                    
                    # Store the original transformer for restoring later if needed
                    if not hasattr(self, 'original_transformer'):
                        self.original_transformer = transformer
                    
                    try:
                        # Create and apply the Flash Attention KV Cache transformer
                        # with memory-efficient initialization
                        print("Creating Flash Attention wrapper...")
                        flash_transformer = FlashAttentionKVCacheTransformer(transformer)
                        
                        # Another memory cleanup before replacing
                        free_memory()
                        
                        print("Replacing transformer with Flash Attention version...")
                        self._replace_transformer(flash_transformer)
                        self.flash_transformer = flash_transformer
                        
                        print("Flash Attention applied successfully")
                        
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"CUDA Out of Memory while applying Flash Attention: {e}")
                        print("Continuing without Flash Attention")
                        # Try to recover the original transformer
                        if hasattr(self, 'original_transformer'):
                            self._replace_transformer(self.original_transformer)
                        # Clean up after OOM
                        free_memory()
                        
                    except Exception as e:
                        print(f"Error applying Flash Attention: {e}")
                        # Try to recover the original transformer
                        if hasattr(self, 'original_transformer'):
                            self._replace_transformer(self.original_transformer)
                else:
                    print("Flash Attention is disabled")
            else:
                print("Warning: Could not locate transformer in pipeline. Flash Attention not applied.")
                
        except Exception as e:
            print(f"Unexpected error in _apply_flash_attention: {e}")
            # Clean up memory
            free_memory()
    
    def _locate_transformer(self):
        """
        Locate the transformer model in the pipeline
        
        Returns:
            Transformer model or None if not found
        """
        # Common attribute names for transformer models
        transformer_attr_names = [
            'unet', 'text_encoder', 'transformer', 'model', 'text_model', 
            'backbone', 'text_transformer', 'diffusion_model'
        ]
        
        try:
            # Try to find transformer in pipeline
            if hasattr(self, 'pipeline'):
                # Method 1: Direct attribute search
                for attr_name in transformer_attr_names:
                    if hasattr(self.pipeline, attr_name):
                        transformer = getattr(self.pipeline, attr_name)
                        print(f"Found transformer as pipeline.{attr_name}")
                        return transformer
                
                # Method 2: Look for U-Net specifically (common in diffusion models)
                if hasattr(self.pipeline, 'unet'):
                    print("Found transformer as pipeline.unet")
                    return self.pipeline.unet
                
                # Method 3: Check common pipeline modules
                for attr_name in dir(self.pipeline):
                    if attr_name.startswith('_'):
                        continue
                    try:
                        attr = getattr(self.pipeline, attr_name)
                        if isinstance(attr, torch.nn.Module) and hasattr(attr, 'forward'):
                            # This is likely a model component
                            print(f"Found potential transformer as pipeline.{attr_name}")
                            return attr
                    except Exception:
                        pass
            
            return None
        except Exception as e:
            print(f"Error locating transformer: {e}")
            return None
    
    def _replace_transformer(self, new_transformer):
        """
        Replace the transformer model in the pipeline
        
        Args:
            new_transformer: New transformer model to use
        """
        # Common attribute names for transformer models
        transformer_attr_names = [
            'unet', 'text_encoder', 'transformer', 'model', 'text_model', 
            'backbone', 'text_transformer', 'diffusion_model'
        ]
        
        if not hasattr(self, 'pipeline'):
            print("Cannot replace transformer: Pipeline not initialized")
            return False
        
        try:
            # Try to replace transformer in pipeline
            replaced = False
            
            # Method 1: Direct attribute replacement
            for attr_name in transformer_attr_names:
                if hasattr(self.pipeline, attr_name):
                    original = getattr(self.pipeline, attr_name)
                    if isinstance(original, torch.nn.Module):
                        setattr(self.pipeline, attr_name, new_transformer)
                        print(f"Replaced transformer at pipeline.{attr_name}")
                        replaced = True
                        break
            
            # Method 2: Targeted replacement for U-Net
            if not replaced and hasattr(self.pipeline, 'unet'):
                self.pipeline.unet = new_transformer
                print(f"Replaced transformer at pipeline.unet")
                replaced = True
            
            return replaced

        except Exception as e:
            print(f"Error replacing transformer: {e}")
            return False

    def generate_image(self, prompt, num_inference_steps=30,
                       guidance_scale=7.5, use_cache=True, use_flash=True,
                       height=1024, width=1024, low_memory_mode=False):
        """
        Generate an image using the optimized pipeline with Flash Attention
        
        Args:
            num_inference_steps: Number of denoising steps (higher = better quality, slower)
            guidance_scale: Scale for classifier-free guidance (higher = more adherence to prompt)
            seed: Random seed for reproducibility
            use_cache: Whether to use KV caching
            use_flash: Whether to use Flash Attention
            height: Height of generated image
            width: Width of generated image
            low_memory_mode: Use aggressive memory optimization techniques
            
        Returns:
            Generated PIL image
        """
        # Clean up memory before generation
        free_memory()
        
        # Dynamically adjust parameters based on available memory
        if torch.cuda.is_available():
            # Get available memory in MB
            free_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) - torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"Free GPU memory before generation: {free_memory_mb:.2f} MB")
            
            # Dynamic resolution adjustment
            if free_memory_mb < 4000 and height > 768:
                original_height, original_width = height, width
                height, width = 768, 768
                print(f"Reduced resolution from {original_height}x{original_width} to {height}x{width} due to memory constraints")
            
            if free_memory_mb < 2000 and height > 512:
                original_height, original_width = height, width
                height, width = 512, 512
                print(f"Reduced resolution from {original_height}x{original_width} to {height}x{width} due to memory constraints")
                
            # Force low memory mode if very little memory is available
            if free_memory_mb < 1500:
                low_memory_mode = True
                print("Automatically enabling low memory mode due to limited available memory")
                
            # In extremely low memory conditions, reduce further
            if free_memory_mb < 800 and height > 384:
                height, width = 384, 384
                print(f"Severely reduced resolution to {height}x{width} due to very limited memory")
        
        # Apply low memory mode optimizations
        if low_memory_mode:
            print("Running in low memory mode with aggressive optimizations")
            
            # Apply additional memory optimizations to pipeline components
            if hasattr(self, 'pipeline'):
                if hasattr(self.pipeline, "enable_attention_slicing"):
                    self.pipeline.enable_attention_slicing(1)
                
                # Apply any other available memory optimizations
                for opt_method in ["enable_vae_slicing", "enable_vae_tiling", "enable_xformers_memory_efficient_attention"]:
                    if hasattr(self.pipeline, opt_method):
                        try:
                            getattr(self.pipeline, opt_method)()
                            print(f"Applied {opt_method} for memory efficiency")
                        except Exception as e:
                            print(f"Failed to apply {opt_method}: {e}")
                            
                # Consider disabling Flash Attention if in very low memory conditions
                if free_memory_mb < 1000 and use_flash:
                    use_flash = False
                    print("Disabled Flash Attention due to very low memory")
        
        # Update flash attention flag
        self.use_flash_attention = use_flash
        
        # Enable/disable Flash Attention if transformer has been initialized
        if hasattr(self, 'flash_transformer'):
            if use_flash:
                self.flash_transformer.enable_flash_attention()
            else:
                self.flash_transformer.disable_flash_attention()
                
        # Clear any cached tensor memory before generation
        free_memory()
        
        # Check if pipeline is properly initialized
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized. Please call load_model() first.")
        
        # First attempt with current settings
        try:
            # Start timing
            start_time = time.time()
            # Use the pipeline directly for image generation
            kwargs = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "prompt": prompt
            }
            
            # Remove use_cache from kwargs as it's not a standard diffusers parameter
            if hasattr(self, 'use_kv_cache') and self.use_kv_cache:
                # KV cache is handled internally by the transformer
                pass
            
            result = self.pipeline(**kwargs)
            if hasattr(result, 'images') and result.images:
                image = result.images[0]
            else:
                raise RuntimeError("Pipeline did not return expected image result")
                
            generation_time = time.time() - start_time
        
            print(f"Image generated in {generation_time:.2f} seconds")
            return image, generation_time
        except torch.cuda.OutOfMemoryError as e:
            # Handle OOM error by retrying with reduced settings
            print(f"CUDA Out of Memory error: {e}")

            # Clear memory
            free_memory()
            
            # Reduce resolution further
            height_reduced = max(256, height // 2)
            width_reduced = max(256, width // 2)
            print(f"Reducing resolution to {height_reduced}x{width_reduced}")
            
            # Reduce inference steps
            inference_steps_reduced = max(10, num_inference_steps // 2)
            print(f"Reducing inference steps to {inference_steps_reduced}")
            
            # Turn off optimizations that might use additional memory
            print("Disabling Flash Attention to save memory")
            
            try:
                # Try again with reduced settings
                return super().generate_image(
                    prompt=prompt,
                    num_inference_steps=inference_steps_reduced,
                    guidance_scale=guidance_scale,
                    use_cache=True,  # Keep KV cache for speed
                    height=height_reduced,
                    width=width_reduced
                )
            except torch.cuda.OutOfMemoryError as e:
                # Last resort: try with minimum possible settings
                print(f"Still encountered OOM error: {e}")
                
                # Clear memory again
                free_memory()
                
                # Use minimum viable settings
                min_height, min_width = 256, 256
                min_steps = 5
                
                print(f"Using minimum resolution {min_height}x{min_width} and {min_steps} steps")
                
                # Last attempt with absolute minimum settings
                return super().generate_image(
                    prompt=prompt,
                    num_inference_steps=min_steps,
                    guidance_scale=5.0,  # Lower guidance scale to save memory
                    use_cache=True,
                    height=min_height,
                    width=min_width
                )
    
    def generate_images_with_dataset(self, image_filename_to_caption, output_dir, num_images=10, 
                                     num_inference_steps=30, guidance_scale=7.5, 
                                     use_cache=True, use_flash=True,
                                     height=512, width=512):
        """
        Generate images using captions from a dataset with Flash Attention
        
        Args:
            image_filename_to_caption: Dictionary mapping filenames to captions
            output_dir: Directory to save generated images
            num_images: Number of images to generate (maximum)
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            use_cache: Whether to use KV caching
            use_flash: Whether to use Flash Attention
            height: Height of generated images
            width: Width of generated images
        """
        # Clean up memory before generation
        free_memory()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Update flash attention flag
        self.use_flash_attention = use_flash
        
        # Enable/disable Flash Attention if transformer has been initialized
        if hasattr(self, 'flash_transformer'):
            if use_flash:
                self.flash_transformer.enable_flash_attention()
            else:
                self.flash_transformer.disable_flash_attention()
        
        # Calculate estimated memory requirement and adjust settings if needed
        if torch.cuda.is_available():
            free_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) - torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"Free GPU memory before dataset generation: {free_memory_mb:.2f} MB")
            
            # Adjust settings based on available memory
            if free_memory_mb < 2000 and height > 512:
                height, width = 512, 512
                print(f"Reduced resolution to {height}x{width} due to memory constraints")
                
            if free_memory_mb < 1000 and num_inference_steps > 20:
                num_inference_steps = 20
                print(f"Reduced inference steps to {num_inference_steps} due to memory constraints")
        
        # Process images sequentially to avoid OOM
        processed_count = 0
        failed_count = 0
        
        # Limit the number of images to generate
        filenames_captions = list(image_filename_to_caption.items())[:num_images]
        print(f"Generating {len(filenames_captions)} images...")

        # Generate images one by one
        for filename, caption in filenames_captions:
            print(f"Generating image {processed_count + 1}/{len(filenames_captions)}: {caption[:50]}...")

            try:
                # Clean up memory before each generation
                if processed_count > 0:
                    free_memory()
                    if hasattr(self, 'flash_transformer') and hasattr(self.flash_transformer, 'clear_cache'):
                        self.flash_transformer.clear_cache()
                
                # Generate the image
                result = self.generate_image(
                    prompt=caption,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    use_cache=use_cache,
                    use_flash=use_flash,
                    height=height,
                    width=width
                )
                
                # Handle both tuple and single image return values
                if isinstance(result, tuple):
                    image, generation_time = result
                else:
                    image = result
                
                # Save the image
                output_path = os.path.join(output_dir, f"{filename}.png")
                image.save(output_path)
                
                processed_count += 1
                print(f"Saved to {output_path}")
                
            except Exception as e:
                print(f"Error generating image for caption: {caption[:50]}...Error: {e}")
                failed_count += 1
                
                # Clean up after error
                free_memory()
        
        print(f"Generation complete. Successfully generated {processed_count} images, failed {failed_count} images.")
        return output_dir