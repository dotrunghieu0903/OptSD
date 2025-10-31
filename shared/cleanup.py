import torch
import gc
import os

def setup_memory_optimizations():
    """Apply memory optimizations to avoid CUDA OOM errors"""
    # PyTorch memory optimizations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
def optimize_for_inference(model):
    """
    Apply inference-time optimizations to reduce memory usage
    
    Args:
        model: PyTorch model to optimize
    
    Returns:
        Optimized model
    """
    # Check if model exists
    if model is None:
        return None
        
    # Convert to eval mode
    model.eval()
    
    # Free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Apply additional optimizations based on model type
    try:
        # Try to use torch.compile if available (requires PyTorch 2.0+)
        if hasattr(torch, 'compile') and callable(torch.compile):
            try:
                print("Attempting to use torch.compile for optimization...")
                compiled_model = torch.compile(model, mode="reduce-overhead")
                print("Model successfully compiled with torch.compile")
                return compiled_model
            except Exception as e:
                print(f"Could not apply torch.compile: {e}")
    except:
        pass
        
    return model

def setup_memory_optimizations_pruning():
    """
    Configure memory optimizations for PyTorch to avoid CUDA OOM errors.
    Applies aggressive memory management for handling large models.
    """
    # Set environment variables for memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    
    # Clear any cached memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Print available GPU memory for debugging
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory = gpu_properties.total_memory / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)
        print(f"GPU: {gpu_properties.name}")
        print(f"Total GPU memory: {total_memory:.2f} GiB")
        print(f"Allocated GPU memory: {allocated_memory:.2f} GiB")
        print(f"Reserved GPU memory: {reserved_memory:.2f} GiB")
        print(f"Free GPU memory: {total_memory - allocated_memory:.2f} GiB")
    
    # Set memory fraction to avoid OOM
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.95)
        
    try:
        # Try to enable memory efficient attention if available
        import importlib
        if importlib.util.find_spec("transformers"):
            import transformers
            if hasattr(transformers, 'utils') and hasattr(transformers.utils, 'logging'):
                transformers.utils.logging.set_verbosity_error()
            
            # Check if we can use BetterTransformer for memory efficiency
            if hasattr(transformers, 'BetterTransformer'):
                print("BetterTransformer is available for memory optimizations")
                
    except (ImportError, AttributeError):
        print("Transformers library not available for additional optimizations")
        
    # Set default tensor type to save memory
    if torch.cuda.is_available() and hasattr(torch, 'set_default_tensor_type'):
        try:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        except:
            pass