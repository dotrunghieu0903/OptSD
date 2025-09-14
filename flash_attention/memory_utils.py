#!/usr/bin/env python3
"""
Memory utilities for Flash Attention module.

This module provides advanced memory optimization utilities specifically designed
for Flash Attention integration with diffusion models.
"""

import gc
import torch
import psutil
import os

def aggressive_memory_cleanup():
    """
    Perform an aggressive memory cleanup to free up both GPU and CPU memory.
    
    This function combines multiple memory cleanup techniques to maximize
    available memory for large model inference.
    """
    # GPU memory cleanup
    if torch.cuda.is_available():
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Clear current device
        device = torch.cuda.current_device()
        torch.cuda.empty_cache()
        
        # Synchronize CUDA streams
        torch.cuda.synchronize(device)
    
    # Python memory cleanup
    gc.collect()
    
    # Additional CPU memory optimization
    if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'memory_summary'):
        # Print memory summary for debugging
        print("Memory summary after cleanup:")
        print(torch.cuda.memory_summary(abbreviated=True))
    
    return get_memory_info()

def get_memory_info():
    """
    Get current memory usage information for both GPU and CPU.
    
    Returns:
        dict: A dictionary containing memory information
    """
    memory_info = {
        'cpu_percent': psutil.virtual_memory().percent,
        'cpu_available_gb': psutil.virtual_memory().available / (1024 ** 3),
        'cpu_total_gb': psutil.virtual_memory().total / (1024 ** 3),
    }
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_info.update({
            'gpu_allocated_gb': torch.cuda.memory_allocated(device) / (1024 ** 3),
            'gpu_reserved_gb': torch.cuda.memory_reserved(device) / (1024 ** 3),
            'gpu_max_allocated_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            'gpu_max_reserved_gb': torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        })
        
        # Try to get the total GPU memory
        try:
            memory_info['gpu_total_gb'] = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            memory_info['gpu_free_gb'] = memory_info['gpu_total_gb'] - memory_info['gpu_allocated_gb']
        except:
            memory_info['gpu_total_gb'] = 'Unknown'
            memory_info['gpu_free_gb'] = 'Unknown'
    
    return memory_info

def print_memory_info(info=None):
    """Print memory information in a readable format"""
    if info is None:
        info = get_memory_info()
    
    print("\n===== MEMORY INFORMATION =====")
    print(f"CPU Memory: {info['cpu_percent']}% used, {info['cpu_available_gb']:.2f}GB free of {info['cpu_total_gb']:.2f}GB")
    
    if 'gpu_allocated_gb' in info:
        print(f"GPU Memory: {info['gpu_allocated_gb']:.2f}GB used, {info['gpu_free_gb']:.2f}GB free of {info['gpu_total_gb']:.2f}GB")
        print(f"GPU Peak Usage: {info['gpu_max_allocated_gb']:.2f}GB")
    
    print("==============================\n")

def optimize_inference_environment():
    """Configure the environment for memory-efficient inference"""
    # Set environment variables for better memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    if torch.cuda.is_available():
        # Optimize CUDA settings for inference
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Use deterministic algorithms only when needed (they can be slower)
        torch.use_deterministic_algorithms(False)
    
    # Return current memory information
    return get_memory_info()