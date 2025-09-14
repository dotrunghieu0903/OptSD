#!/usr/bin/env python3
"""
Flash Attention Integration for Diffusion Models

This module integrates Flash Attention with the KV caching mechanism of the diffusion models.
It enhances the attention mechanism performance while maintaining compatibility with the
existing optimization techniques.
"""

import os
import torch
import sys
import gc

# Configure PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attention.flash_attn_module import FlashAttentionModule, FlashMHA

def free_memory():
    """Free unused GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ======================================================================
# FLASH ATTENTION KV CACHE TRANSFORMER
# ======================================================================

class FlashAttentionKVCacheTransformer(torch.nn.Module):
    """Enhanced Transformer model with Flash Attention and KV caching
    
    This class wraps a transformer model and adds both Flash Attention and KV caching
    to speed up the image generation process. It uses Flash Attention for efficient
    attention computation and KV caching to reuse previous computations.
    """
    
    def __init__(self, transformer, memory_efficient=True):
        super().__init__()
        # Store the original transformer
        self.transformer = transformer
        
        # Initialize KV cache
        self.kv_cache = {}
        
        # Flags
        self.use_kv_cache = False
        self.use_flash_attention = True
        self.memory_efficient = memory_efficient
        
        # Free memory before intensive operations
        if self.memory_efficient:
            free_memory()
        
        # Copy device and dtype from original transformer
        self.device = getattr(transformer, 'device', torch.device('cuda'))
        self.dtype = getattr(transformer, 'dtype', None)
        
        # Fallback for dtype
        if self.dtype is None:
            for param in transformer.parameters():
                if param.dtype:
                    self.dtype = param.dtype
                    break
        
        # Copy config from original transformer
        if hasattr(transformer, 'config'):
            self.config = transformer.config
        
        # Initialize Flash Attention module
        self.flash_attn = FlashAttentionModule(use_triton=True)
        
        # Apply Flash Attention and KV caching to attention blocks
        print("Inspecting transformer structure to apply Flash Attention and KV caching...")
        self._inspect_structure()
        self._apply_flash_attention_to_attention_blocks()
    
    def _inspect_structure(self, max_depth=3):
        """Inspect the transformer structure to identify attention modules"""
        self.attention_patterns = []
        self.attention_modules = {}
        
        # Free memory before the operation if in memory-efficient mode
        if self.memory_efficient:
            free_memory()
        
        def _inspect_module(module, name_prefix="", depth=0):
            if depth > max_depth:
                return
                
            for name, child in module.named_children():
                full_name = f"{name_prefix}.{name}" if name_prefix else name
                
                # Check if this might be an attention module
                if self._is_attention_module(full_name, child):
                    print(f"Found potential attention module: {full_name}")
                    self.attention_modules[full_name] = child
                    self.attention_patterns.append((full_name, type(child).__name__))
                
                # Recursively inspect children
                _inspect_module(child, full_name, depth + 1)
                
                # Periodically free memory in memory-efficient mode
                if self.memory_efficient and depth == 0 and torch.cuda.is_available():
                    if torch.cuda.memory_allocated() > 0.9 * torch.cuda.get_device_properties(0).total_memory:
                        free_memory()
        
        try:
            _inspect_module(self.transformer)
            print(f"Found {len(self.attention_modules)} potential attention modules")
        except torch.cuda.OutOfMemoryError:
            print("Out of memory during structure inspection. Using simplified detection.")
            free_memory()
            self.attention_modules = {}
            self.attention_patterns = []
        
        # If no attention modules found, try alternative detection
        if not self.attention_modules:
            self._try_alternative_attention_detection()
    
    def _is_attention_module(self, name, module):
        """Check if a module is likely an attention module based on name and attributes"""
        attention_keywords = ['attn', 'attention', 'self_attn', 'cross_attn']
        
        # Check name for attention keywords
        name_match = any(keyword in name.lower() for keyword in attention_keywords)
        
        # Check if module has typical attention attributes
        has_attention_attributes = (
            hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj')
        ) or (
            hasattr(module, 'query') and hasattr(module, 'key') and hasattr(module, 'value')
        )
        
        return name_match or has_attention_attributes
    
    def _try_alternative_attention_detection(self):
        """Try alternative methods to detect attention modules"""
        print("Trying alternative attention detection methods...")
        
        # Method 1: Look for forward methods that accept query, key, value parameters
        import inspect
        
        def _check_module_signature(module, name_prefix="", depth=0):
            if depth > 3:
                return
                
            for name, child in module.named_children():
                full_name = f"{name_prefix}.{name}" if name_prefix else name
                
                if hasattr(child, 'forward') and inspect.ismethod(child.forward):
                    sig = inspect.signature(child.forward)
                    params = list(sig.parameters.keys())
                    if len(params) >= 4 and all(p in params for p in ['self', 'query']):
                        if ('key' in params and 'value' in params) or ('k' in params and 'v' in params):
                            print(f"Found attention by signature: {full_name}")
                            self.attention_modules[full_name] = child
                            self.attention_patterns.append((full_name, type(child).__name__))
                
                # Recursively check children
                _check_module_signature(child, full_name, depth + 1)
        
        _check_module_signature(self.transformer)
        
        if not self.attention_modules:
            print("Warning: Could not find attention modules. Flash Attention and KV caching will not be applied.")
    
    def _apply_flash_attention_to_attention_blocks(self):
        """Replace standard attention with Flash Attention + KV caching in attention blocks"""
        print(f"Applying Flash Attention and KV caching to {len(self.attention_modules)} modules...")
        
        for name, module in self.attention_modules.items():
            # Store the original forward method
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
            
            # Store module name for reference
            module.name = name
            
            # Replace forward method with Flash Attention + KV cache version
            module.forward = self._make_flash_kv_cached_attn_forward(module)
            print(f"Applied Flash Attention + KV caching to: {name}")
    
    def _make_flash_kv_cached_attn_forward(self, module):
        """Create a forward function with both Flash Attention and KV caching"""
        
        def flash_kv_cached_forward(hidden_states, *args, **kwargs):
            # Get timestep and use_cache from kwargs
            timestep = kwargs.get('timestep', 0)
            cache_key = f"{module.name}_{timestep}"
            use_cache = kwargs.get('use_cache', self.use_kv_cache)
            use_flash = getattr(self, 'use_flash_attention', True)
            
            # Check if this is the first run or caching is disabled
            if not use_cache or cache_key not in self.kv_cache:
                # For first run, we calculate Q, K, V and apply Flash Attention
                
                # Method 1: Modules with separate q_proj, k_proj, v_proj (common pattern)
                if all(hasattr(module, attr) for attr in ["q_proj", "k_proj", "v_proj"]):
                    # Get Q, K, V projections
                    q = module.q_proj(hidden_states)
                    k = module.k_proj(hidden_states)
                    v = module.v_proj(hidden_states)
                    
                    if use_cache:
                        # Store K, V in cache for future use
                        self.kv_cache[cache_key] = (k, v)
                    
                    # Try to reshape for attention calculation
                    try:
                        batch_size, seq_len, _ = q.size()
                        head_dim = q.size(-1) // getattr(module, "num_heads", 8)
                        num_heads = getattr(module, "num_heads", 8)
                        
                        # Reshape for attention calculation [batch, seq_len, embed_dim] -> [batch, heads, seq_len, head_dim]
                        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                        
                        if use_flash:
                            # Use Flash Attention
                            try:
                                is_causal = getattr(module, "causal", False)
                                attn_output = self.flash_attn(q, k, v, is_causal=is_causal)
                            except Exception as e:
                                print(f"Flash attention failed: {e}. Using standard attention.")
                                # Calculate attention scores
                                attention_scores = torch.matmul(q, k.transpose(-1, -2))
                                attention_scores = attention_scores / (head_dim ** 0.5)
                                
                                # Apply softmax
                                attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
                                attn_output = torch.matmul(attention_probs, v)
                        else:
                            # Standard attention
                            attention_scores = torch.matmul(q, k.transpose(-1, -2))
                            attention_scores = attention_scores / (head_dim ** 0.5)
                            
                            # Apply softmax
                            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
                            attn_output = torch.matmul(attention_probs, v)
                        
                        # Reshape back to [batch, seq_len, embed_dim]
                        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                        
                        # Apply output projection if available
                        if hasattr(module, "out_proj"):
                            attn_output = module.out_proj(attn_output)
                        
                        return attn_output
                    except Exception as e:
                        print(f"Error in custom attention: {e}. Using original forward.")
                        return module.original_forward(hidden_states, *args, **kwargs)
                    
                # If not standard pattern, use original forward
                else:
                    return module.original_forward(hidden_states, *args, **kwargs)
            else:
                # If KV cache is available, use cached K, V values
                k, v = self.kv_cache[cache_key]
                
                # Method 1: Modules with q_proj
                if hasattr(module, "q_proj"):
                    # Calculate query for current token
                    q = module.q_proj(hidden_states)
                    
                    try:
                        # Prepare input data
                        batch_size, seq_len, _ = q.size()
                        head_dim = q.size(-1) // getattr(module, "num_heads", 8)
                        num_heads = getattr(module, "num_heads", 8)
                        
                        # Reshape for attention calculation
                        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                        k_shape = k.size()
                        v_shape = v.size()
                        
                        # Ensure K and V are properly shaped
                        if len(k_shape) == 3:
                            k = k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
                            v = v.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
                        
                        if use_flash:
                            # Use Flash Attention
                            try:
                                is_causal = getattr(module, "causal", False)
                                attn_output = self.flash_attn(q, k, v, is_causal=is_causal)
                            except Exception as e:
                                print(f"Flash attention failed: {e}. Using standard attention.")
                                # Calculate attention scores
                                attention_scores = torch.matmul(q, k.transpose(-1, -2))
                                attention_scores = attention_scores / (head_dim ** 0.5)
                                
                                # Apply softmax
                                attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
                                attn_output = torch.matmul(attention_probs, v)
                        else:
                            # Standard attention
                            attention_scores = torch.matmul(q, k.transpose(-1, -2))
                            attention_scores = attention_scores / (head_dim ** 0.5)
                            
                            # Apply softmax
                            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
                            attn_output = torch.matmul(attention_probs, v)
                        
                        # Reshape back to [batch, seq_len, embed_dim]
                        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                        
                        # Apply output projection if available
                        if hasattr(module, "out_proj"):
                            attn_output = module.out_proj(attn_output)
                        
                        return attn_output
                    except Exception as e:
                        print(f"Error in custom attention with cache: {e}. Using original forward.")
                        return module.original_forward(hidden_states, *args, **kwargs)
                else:
                    # Use original forward if attention structure is not suitable
                    return module.original_forward(hidden_states, *args, **kwargs)
        
        return flash_kv_cached_forward
    
    def clear_cache(self):
        """Clear the KV cache and free memory"""
        self.kv_cache = {}
        
        # Also free GPU memory
        if self.memory_efficient:
            free_memory()
    
    def enable_kv_caching(self):
        """Enable KV caching"""
        self.use_kv_cache = True
    
    def disable_kv_caching(self):
        """Disable KV caching"""
        self.use_kv_cache = False
    
    def enable_flash_attention(self):
        """Enable Flash Attention"""
        self.use_flash_attention = True
    
    def disable_flash_attention(self):
        """Disable Flash Attention"""
        self.use_flash_attention = False
    
    def to(self, *args, **kwargs):
        """Override to() method to support device movement"""
        result = super().to(*args, **kwargs)
        
        # Update device information
        if args and isinstance(args[0], (str, torch.device)):
            self.device = torch.device(args[0])
        elif 'device' in kwargs:
            self.device = torch.device(kwargs['device'])
        
        # Update dtype information
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        
        # Move transformer
        if hasattr(self, 'transformer'):
            self.transformer.to(*args, **kwargs)
        
        return result
    
    def forward(self, *args, **kwargs):
        """Forward method with Flash Attention and KV caching support"""
        # Check if use_cache is provided and handle it
        if 'use_cache' in kwargs:
            self.use_kv_cache = kwargs.pop('use_cache')
        
        # Check if use_flash is provided
        if 'use_flash' in kwargs:
            self.use_flash_attention = kwargs.pop('use_flash')
        
        try:
            # Perform forward through replaced transformer
            return self.transformer(*args, **kwargs)
        except Exception as e:
            print(f"Error in FlashAttentionKVCacheTransformer.forward: {e}")
            print(f"Args types: {[type(arg) for arg in args]}")
            print(f"Kwargs keys: {list(kwargs.keys())}")
            
            # Try with filtered kwargs
            try:
                kwargs_filtered = {k: v for k, v in kwargs.items() 
                                if k not in ['return_dict', 'output_attentions']}
                return self.transformer(*args, **kwargs_filtered)
            except Exception as e2:
                print(f"Error with filtered kwargs: {e2}")
                
                # Try with only args
                try:
                    return self.transformer(*args)
                except Exception as e3:
                    print(f"Error with only args: {e3}")
                    raise e  # Throw original exception