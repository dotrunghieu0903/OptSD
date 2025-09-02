#!/usr/bin/env python3

import argparse
import gc
import time
import json
import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import DiffusionPipeline
from huggingface_hub import login
import threading

# Add the project root to the path to access shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import calculate_clip_score, calculate_fid, calculate_lpips, calculate_psnr_resized, compute_image_reward
from resizing_image import resize_images
from shared.resources_monitor import generate_image_and_monitor, monitor_vram, write_generation_metadata_to_file

# A list to store VRAM usage samples (to be used by monitor_vram_detailed)
vram_samples = []
stop_monitoring = threading.Event()

# Memory utilities for optimization
def setup_memory_optimizations():
    """Apply memory optimizations to avoid CUDA OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class KVCacheTransformer(torch.nn.Module):
    """Transformer model with Key-Value caching for transformer blocks
    
    SANA model sử dụng transformer thay vì UNet như trong các mô hình Diffusion thông thường.
    Class này wrap transformer và thêm KV caching để tăng tốc quá trình sinh ảnh.
    """
    
    def __init__(self, transformer):
        super().__init__()
        # Copy transformer gốc
        self.transformer = transformer
        
        # Khởi tạo bộ nhớ cache KV
        self.kv_cache = {}
        
        # Cờ báo hiệu có sử dụng KV cache hay không
        self.use_kv_cache = False
        
        # Copy device và dtype từ transformer gốc nếu có
        self.device = getattr(transformer, 'device', torch.device('cuda'))
        self.dtype = getattr(transformer, 'dtype', None)
        
        # Fallback to transformer parameters
        if self.dtype is None:
            # Try to infer dtype from transformer parameters
            for param in transformer.parameters():
                if param.dtype:
                    self.dtype = param.dtype
                    break
        
        # Copy config từ transformer gốc hoặc tạo một config mới nếu không có
        if hasattr(transformer, 'config'):
            self.config = transformer.config
        else:
            # Tạo một config tạm thời để tránh lỗi
            from types import SimpleNamespace
            self.config = SimpleNamespace()
            
            # Thêm các thuộc tính cần thiết vào config
            if hasattr(transformer, 'dim'):
                self.config.hidden_size = transformer.dim
            elif hasattr(transformer, 'hidden_size'):
                self.config.hidden_size = transformer.hidden_size
            else:
                self.config.hidden_size = 768  # Giá trị mặc định thông thường
                
            # Thêm các thuộc tính khác nếu cần
            self.config.model_type = "transformer"
            
            # Sao chép thêm các thuộc tính từ transformer gốc
            for attr in dir(transformer):
                if not attr.startswith('_') and not hasattr(self.config, attr):
                    try:
                        value = getattr(transformer, attr)
                        if not callable(value) and not isinstance(value, torch.nn.Module):
                            setattr(self.config, attr, value)
                    except:
                        pass
            
            print("Created fallback config for transformer")
        
        # Lưu thông tin về cấu trúc transformer để debug
        self.transformer_structure = self._inspect_structure(transformer)
        
    def _inspect_structure(self, module, prefix="", max_depth=3, current_depth=0):
        """Kiểm tra cấu trúc của module để hiểu rõ hơn về transformer"""
        if current_depth >= max_depth:
            return {"type": str(type(module).__name__), "truncated": True}
            
        result = {"type": str(type(module).__name__)}
        
        if hasattr(module, "named_children"):
            children = {}
            has_children = False
            
            for name, child in module.named_children():
                has_children = True
                children[name] = self._inspect_structure(
                    child, 
                    prefix=f"{prefix}.{name}", 
                    max_depth=max_depth, 
                    current_depth=current_depth+1
                )
                
            if has_children:
                result["children"] = children
                
        return result
        
    def _apply_kv_caching_to_attention_blocks(self):
        """Áp dụng KV caching cho các block attention trong transformer"""
        # Tìm kiếm tất cả attention layers
        attention_layers_count = 0
        
        for name, module in self.transformer.named_modules():
            # Tìm các module attention dựa trên tên hoặc cấu trúc
            if self._is_attention_module(name, module):
                if hasattr(module, "forward"):
                    # Lưu hàm forward gốc
                    original_forward = module.forward
                    module.name = name
                    module.original_forward = original_forward
                    
                    # Thay thế bằng phiên bản có cache
                    module.forward = self._make_kv_cached_attn_forward(module)
                    attention_layers_count += 1
                    print(f"Applied KV caching to {name}")
        
        if attention_layers_count == 0:
            print("Warning: No attention layers found for KV caching")
            # Thử tìm kiếm phương pháp khác để áp dụng KV caching
            self._try_alternative_attention_detection()
        else:
            print(f"Successfully applied KV caching to {attention_layers_count} attention layers")
    
    def _is_attention_module(self, name, module):
        """Xác định xem module có phải là attention layer không"""
        # Phương pháp 1: Dựa trên tên
        if any(pattern in name.lower() for pattern in ["attn", "attention"]):
            return True
            
        # Phương pháp 2: Dựa trên thuộc tính đặc trưng của attention
        attn_attributes = ["q_proj", "k_proj", "v_proj", "out_proj"]
        if all(hasattr(module, attr) for attr in attn_attributes):
            return True
            
        # Phương pháp 3: Dựa trên cấu trúc dữ liệu
        if hasattr(module, "forward") and any(param.name in ["query", "key", "value"] for param in module.parameters()):
            return True
            
        return False
    
    def _try_alternative_attention_detection(self):
        """Thử các phương pháp thay thế để tìm và áp dụng KV caching"""
        print("Trying alternative methods to detect attention layers...")
        
        # Phương pháp 1: Tìm kiếm theo cấu trúc các phương thức
        found_layers = []
        for name, module in self.transformer.named_modules():
            if hasattr(module, "forward") and "attention" in str(module.forward).lower():
                found_layers.append((name, module))
                
        # Phương pháp 2: Xem xét các layer có tên gợi ý attention
        for name, module in self.transformer.named_modules():
            if any(hint in name.lower() for hint in ["self", "cross", "mha", "mhsa"]):
                if (name, module) not in found_layers:
                    found_layers.append((name, module))
        
        # Áp dụng KV caching cho các layer tìm được
        for name, module in found_layers:
            original_forward = module.forward
            module.name = name
            module.original_forward = original_forward
            module.forward = self._make_kv_cached_attn_forward(module)
            print(f"Applied KV caching to {name} using alternative detection")
    
    def _make_kv_cached_attn_forward(self, module):
        """Tạo hàm forward có KV caching cho các attention block
        
        Logic chính:
        1. Lần đầu tiên sinh token: Tính và lưu giá trị KV
        2. Các lần tiếp theo: Sử dụng lại giá trị KV đã tính, chỉ tính Q mới
        3. Tính toán attention hiệu quả: Sử dụng K, V đã cache với Q mới
        """
        def cached_forward(hidden_states, *args, **kwargs):
            # Lấy thông tin timestep từ kwargs nếu có
            timestep = kwargs.get('timestep', 0)
            cache_key = f"{module.name}_{timestep}"
            use_cache = kwargs.get('use_cache', False)
            
            # Kiểm tra xem đây có phải lần đầu chạy hoặc cache bị tắt không
            if not use_cache or cache_key not in self.kv_cache:
                # Gọi hàm forward gốc
                output = module.original_forward(hidden_states, *args, **kwargs)
                
                # Lưu giá trị KV vào cache nếu module có các thành phần cần thiết
                if use_cache:
                    # 2. Xử lý KV caching - Phương pháp 1: Các module có q_proj, k_proj, v_proj riêng biệt
                    if all(hasattr(module, attr) for attr in ["q_proj", "k_proj", "v_proj"]):
                        key = module.k_proj(hidden_states)
                        value = module.v_proj(hidden_states)
                        self.kv_cache[cache_key] = (key, value)
                    
                    # Phương pháp 2: Module có hàm tạo key, value
                    elif hasattr(module, "get_key_value"):
                        key, value = module.get_key_value(hidden_states)
                        self.kv_cache[cache_key] = (key, value)
                        
                    # Phương pháp 3: Module có thuộc tính key, value
                    elif hasattr(module, "key") and hasattr(module, "value"):
                        try:
                            key = module.key(hidden_states)
                            value = module.value(hidden_states)
                            self.kv_cache[cache_key] = (key, value)
                        except Exception as e:
                            print(f"Failed to cache key-value: {e}")
            else:
                # Lấy giá trị KV từ cache
                key, value = self.kv_cache[cache_key]
                
                # Tính query cho token hiện tại
                if hasattr(module, "q_proj"):
                    query = module.q_proj(hidden_states)
                    
                    # 3. Tính toán attention hiệu quả
                    # Điều chỉnh tùy theo kiến trúc attention của SANA
                    try:
                        # Chuẩn bị dữ liệu đầu vào
                        batch_size, seq_len, _ = query.size()
                        head_dim = query.size(-1) // getattr(module, "num_heads", 8)
                        
                        # Reshape để tính attention
                        query = query.view(batch_size, seq_len, -1, head_dim)
                        key = key.view(batch_size, -1, query.size(2), head_dim)
                        value = value.view(batch_size, -1, query.size(2), head_dim)
                        
                        # Tính attention scores
                        attention_scores = torch.matmul(query, key.transpose(-1, -2))
                        attention_scores = attention_scores / (head_dim ** 0.5)
                        
                        # Áp dụng softmax
                        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
                        
                        # Tính context layer
                        context_layer = torch.matmul(attention_probs, value)
                        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                        
                        # Reshape lại đầu ra
                        output = context_layer.view(batch_size, seq_len, -1)
                        
                        # Áp dụng output projection nếu có
                        if hasattr(module, "out_proj"):
                            output = module.out_proj(output)
                    except Exception as e:
                        # Nếu gặp lỗi, sử dụng forward gốc
                        print(f"Error in custom attention: {e}. Using original forward.")
                        output = module.original_forward(hidden_states, *args, **kwargs)
                else:
                    # Sử dụng forward gốc nếu không có cấu trúc attention phù hợp
                    output = module.original_forward(hidden_states, *args, **kwargs)
            
            return output
        
        return cached_forward
    
    def clear_cache(self):
        """Xóa toàn bộ KV cache"""
        self.kv_cache = {}
        print("KV cache cleared")
    
    def enable_kv_caching(self):
        """Bật tính năng KV caching trong transformer"""
        self._apply_kv_caching_to_attention_blocks()
        print("KV caching enabled for Transformer model")
    
    def to(self, *args, **kwargs):
        """Custom to() method to handle device and dtype"""
        # Update internal device and dtype attributes
        if 'device' in kwargs:
            self.device = kwargs['device']
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
            
        # Call parent to() method to move module parameters
        result = super().to(*args, **kwargs)
        
        # Also move the wrapped transformer if possible
        try:
            self.transformer.to(*args, **kwargs)
        except Exception as e:
            print(f"Warning: Could not move wrapped transformer: {e}")
            
        return result
    
    def forward(self, *args, **kwargs):
        """Forward method với hỗ trợ KV caching"""
        # Kiểm tra xem use_cache có được cung cấp không và xử lý nó
        if 'use_cache' in kwargs:
            self.use_kv_cache = kwargs.pop('use_cache')
        
        # Lưu tất cả các tham số đầu vào cho việc debug
        if not hasattr(self, '_last_inputs'):
            self._last_inputs = {'args': args, 'kwargs': kwargs}
        
        # Sử dụng thuộc tính use_kv_cache nếu có
        use_cache = getattr(self, "use_kv_cache", False)
        
        # Kiểm tra xem transformer có hỗ trợ use_cache không
        supports_use_cache = False
        if use_cache:
            # Kiểm tra signature của phương thức forward
            import inspect
            if hasattr(self.transformer, 'forward') and inspect.isfunction(self.transformer.forward):
                sig = inspect.signature(self.transformer.forward)
                supports_use_cache = 'use_cache' in sig.parameters
            
            # Hoặc kiểm tra từ các thuộc tính khác nếu có
            model_config = getattr(self.transformer, 'config', None)
            if model_config and hasattr(model_config, 'use_cache'):
                supports_use_cache = True
        
        try:
            # Thử với use_cache nếu phù hợp và được hỗ trợ
            if use_cache and supports_use_cache:
                try:
                    # Clone kwargs để không ảnh hưởng đến tham số gốc
                    new_kwargs = dict(kwargs)
                    new_kwargs["use_cache"] = True
                    return self.transformer(*args, **new_kwargs)
                except Exception as e:
                    print(f"Warning: KV cache parameter supported but failed: {e}")
                    # Tiếp tục thử không có use_cache
            elif use_cache and not supports_use_cache:
                # Tốt nhất là không hiển thị thông báo này nếu KV cache không được hỗ trợ
                # vì đây là trường hợp bình thường với nhiều model
                pass
            
            # Thực hiện forward thông qua transformer đã được thay thế
            return self.transformer(*args, **kwargs)
        except Exception as e:
            print(f"Error in KVCacheTransformer.forward: {e}")
            print(f"Args types: {[type(arg) for arg in args]}")
            print(f"Kwargs keys: {list(kwargs.keys())}")
            
            # Thử lại với các điều chỉnh khác nhau
            try:
                # Thử loại bỏ các tham số có thể gây lỗi
                kwargs_filtered = {k: v for k, v in kwargs.items() 
                                if k not in ['return_dict', 'output_attentions', 'use_cache']}
                return self.transformer(*args, **kwargs_filtered)
            except Exception as e2:
                print(f"Error with filtered kwargs: {e2}")
                
                # Thử với chỉ args không có kwargs
                try:
                    return self.transformer(*args)
                except Exception as e3:
                    print(f"Error with only args: {e3}")
                    raise e  # Ném ngoại lệ ban đầu


class OptimizedSANAPipeline:
    """Pipeline for SANA model with KV caching optimization
    
    Class này khởi tạo và quản lý SANA pipeline với tối ưu KV caching.
    Thay vì sử dụng UNet như mô hình Diffusion khác, SANA sử dụng transformer.
    """
    
    def __init__(self, model_path="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"):
        """Initialize the pipeline with KV caching"""
        self.model_path = model_path
        self.pipeline = None
        self.kv_cached_transformer = None
        self.pipeline_type = None
        self.load_model()
        
    def load_model(self):
        """Tải SANA model và áp dụng KV caching"""
        print(f"Loading model from {self.model_path}...")
        
        # Áp dụng tối ưu bộ nhớ
        setup_memory_optimizations()
        
        try:
            # Tải pipeline
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            )
            
            # Lưu thông tin loại pipeline
            self.pipeline_type = type(self.pipeline).__name__
            print(f"Loaded pipeline: {self.pipeline_type}")
            
            # Kiểm tra và cảnh báo nếu là SCM model
            if ("scm" in self.model_path.lower() or 
                "consistency" in self.pipeline_type.lower() or
                hasattr(self.pipeline, "is_scm")):
                print("\n============================================")
                print("WARNING: Detected SCM (Stable Consistency Model)")
                print("SCM models require exactly 2 inference steps")
                print("Any other value will be automatically adjusted to 2")
                print("============================================\n")
            
            # Chuyển đến GPU
            self.pipeline = self.pipeline.to("cuda")
            
            # Kiểm tra cấu trúc pipeline
            print("\n=== Pipeline components ===")
            if hasattr(self.pipeline, 'components'):
                print("Available components:", list(self.pipeline.components.keys()))
                
            # Tìm thành phần transformer
            transformer = self._locate_transformer()
            
            # Tạo transformer có KV cache
            self.kv_cached_transformer = KVCacheTransformer(transformer)
            
            # Lấy device và dtype từ transformer nếu có, hoặc sử dụng giá trị mặc định
            device = transformer.device if hasattr(transformer, 'device') else torch.device('cuda')
            dtype = transformer.dtype if hasattr(transformer, 'dtype') else None
            
            # Chuyển transformer sang device và dtype phù hợp
            if dtype is not None:
                self.kv_cached_transformer.to(device=device, dtype=dtype)
            else:
                self.kv_cached_transformer.to(device=device)
            
            # Bật KV caching
            self.kv_cached_transformer.enable_kv_caching()
            
            # Thay thế transformer trong pipeline
            self._replace_transformer(transformer)
            
            print(f"Model loaded and KV caching enabled")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _locate_transformer(self):
        """Tìm thành phần transformer trong pipeline"""
        # Phương pháp 1: Kiểm tra thuộc tính trực tiếp
        if hasattr(self.pipeline, 'transformer'):
            transformer = self.pipeline.transformer
            print("Found transformer at pipeline.transformer")
            return transformer
        
        # Phương pháp 2: Kiểm tra trong components
        elif hasattr(self.pipeline, 'components') and 'transformer' in self.pipeline.components:
            transformer = self.pipeline.components['transformer']
            print("Found transformer at pipeline.components['transformer']")
            return transformer
        
        # 4. Sửa lỗi chung - Thông tin chi tiết khi không tìm thấy transformer
        else:
            print("\n=== Pipeline structure debug information ===")
            print(f"Pipeline type: {self.pipeline_type}")
            print("\nPipeline attributes:")
            
            # Liệt kê các thuộc tính của pipeline
            for attr in dir(self.pipeline):
                if not attr.startswith("_"):
                    try:
                        attr_value = getattr(self.pipeline, attr)
                        print(f"- {attr} ({type(attr_value).__name__})")
                        
                        # Hiển thị thông tin chi tiết về components và model
                        if attr == "components" and isinstance(attr_value, dict):
                            print("  Components keys:", list(attr_value.keys()))
                        elif attr == "model" and attr_value is not None:
                            print("  Model attributes:")
                            for model_attr in dir(attr_value):
                                if not model_attr.startswith("_"):
                                    try:
                                        model_attr_value = getattr(attr_value, model_attr)
                                        print(f"    - {model_attr} ({type(model_attr_value).__name__})")
                                    except:
                                        print(f"    - {model_attr} (error getting type)")
                    except:
                        print(f"- {attr} (error accessing)")
            
            # Tìm kiếm các thuộc tính có thể chứa transformer
            potential_transformer_containers = []
            for attr in dir(self.pipeline):
                if not attr.startswith("_"):
                    try:
                        attr_value = getattr(self.pipeline, attr)
                        if "transform" in str(attr_value).lower():
                            potential_transformer_containers.append(attr)
                    except:
                        pass
            
            if potential_transformer_containers:
                print("\nAttributes that might contain transformer:", potential_transformer_containers)
                
                # Thử lấy transformer từ thuộc tính đầu tiên
                if potential_transformer_containers:
                    try:
                        transformer = getattr(self.pipeline, potential_transformer_containers[0])
                        print(f"Using {potential_transformer_containers[0]} as transformer")
                        return transformer
                    except:
                        pass
            
            raise AttributeError("Could not locate transformer in the SANA pipeline structure")
    
    def _replace_transformer(self, original_transformer):
        """Thay thế transformer gốc bằng phiên bản có KV cache"""
        # Lưu transformer gốc để có thể khôi phục nếu cần
        if not hasattr(self, 'original_transformer') or self.original_transformer is None:
            self.original_transformer = original_transformer
            print("Saved original transformer reference for fallback")
        
        # Lưu vị trí thay thế để có thể khôi phục chính xác
        self.transformer_locations = []
        replaced = False
        
        # Phương pháp 1: Thay thế qua thuộc tính trực tiếp
        if hasattr(self.pipeline, 'transformer'):
            self.pipeline.transformer = self.kv_cached_transformer
            self.transformer_locations.append(('direct', 'transformer'))
            print("Replaced transformer at pipeline.transformer")
            replaced = True
            
        # Phương pháp 2: Thay thế trong components
        if hasattr(self.pipeline, 'components') and 'transformer' in self.pipeline.components:
            self.pipeline.components['transformer'] = self.kv_cached_transformer
            self.transformer_locations.append(('components', 'transformer'))
            print("Replaced transformer at pipeline.components['transformer']")
            replaced = True
            
        # Tìm và thay thế trong tất cả các vị trí có thể
        # 4. Sửa lỗi chung - Tìm thêm thông tin để thay thế
        for attr in dir(self.pipeline):
            if attr.startswith("_") or attr in ['transformer', 'components']:
                continue
                
            try:
                attr_value = getattr(self.pipeline, attr)
                if attr_value is original_transformer:
                    setattr(self.pipeline, attr, self.kv_cached_transformer)
                    self.transformer_locations.append(('attr', attr))
                    print(f"Replaced transformer at pipeline.{attr}")
                    replaced = True
            except Exception as e:
                pass
        
        # Tìm kiếm trong các thuộc tính lồng nhau
        for attr in dir(self.pipeline):
            if attr.startswith("_"):
                continue
                
            try:
                container = getattr(self.pipeline, attr)
                if hasattr(container, "__dict__"):
                    for sub_attr, sub_value in container.__dict__.items():
                        if sub_value is original_transformer:
                            setattr(container, sub_attr, self.kv_cached_transformer)
                            self.transformer_locations.append(('nested', attr, sub_attr))
                            print(f"Replaced transformer at pipeline.{attr}.{sub_attr}")
                            replaced = True
            except Exception as e:
                pass
                
        # Kiểm tra kết quả
        if replaced:
            print(f"Successfully replaced transformer at {len(self.transformer_locations)} locations")
        else:
            print("Warning: Could not find exact location to replace transformer")
            print("Will still use KV caching through the original reference")
            
        # Thêm phương thức để khôi phục transformer gốc
        def restore_original_transformer(self):
            """Khôi phục transformer gốc"""
            if hasattr(self, 'original_transformer') and self.original_transformer is not None:
                for loc_type, *loc_path in self.transformer_locations:
                    try:
                        if loc_type == 'direct':
                            self.pipeline.transformer = self.original_transformer
                        elif loc_type == 'components':
                            self.pipeline.components['transformer'] = self.original_transformer
                        elif loc_type == 'attr':
                            setattr(self.pipeline, loc_path[0], self.original_transformer)
                        elif loc_type == 'nested':
                            container = getattr(self.pipeline, loc_path[0])
                            setattr(container, loc_path[1], self.original_transformer)
                    except Exception as e:
                        print(f"Error restoring transformer at {loc_type} {loc_path}: {e}")
                print("Restored original transformer")
            else:
                print("No original transformer to restore")
                
        # Thêm phương thức vào instance
        self.restore_original_transformer = restore_original_transformer.__get__(self)
    
    def generate_image(self, prompt, negative_prompt="", num_inference_steps=30, 
                      guidance_scale=7.5, seed=None, use_cache=True, height=1024, width=1024):
        """Generate an image using the KV cached pipeline"""
        # Đặt seed nếu được cung cấp
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Xóa KV cache trước khi bắt đầu sinh ảnh mới
        if use_cache and hasattr(self.kv_cached_transformer, 'clear_cache'):
            self.kv_cached_transformer.clear_cache()
            print("KV cache cleared for new image generation")
        
        # Bắt đầu tính thời gian
        start_time = time.time()
        
        # 3. Xử lý tương thích với API - Điều chỉnh tham số theo loại pipeline
        print(f"Using pipeline type: {self.pipeline_type}")
        
        # Kiểm tra loại của pipeline
        pipeline_type = self.pipeline_type.lower()
        is_sana_sprint = "sanasprint" in pipeline_type
        
        # Chuẩn bị các tham số cơ bản
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
        }
        
        # Thêm negative_prompt chỉ khi được hỗ trợ
        if negative_prompt and not is_sana_sprint:
            kwargs["negative_prompt"] = negative_prompt
        
        # Đặt trạng thái use_kv_cache cho transformer
        if hasattr(self.kv_cached_transformer, "use_kv_cache"):
            prev_use_cache = self.kv_cached_transformer.use_kv_cache
            self.kv_cached_transformer.use_kv_cache = use_cache
            print(f"Set KV caching state: {use_cache}")
        
        # Kiểm tra nếu đang sử dụng SCM model
        is_scm_model = (
            "scm" in self.model_path.lower() or 
            hasattr(self.pipeline, "is_scm") or
            (hasattr(self.pipeline, "components") and any("scm" in str(v).lower() for v in self.pipeline.components.values())) or
            "scm" in self.pipeline_type.lower() or
            "consistency" in self.pipeline_type.lower()
        )
        
        # Nếu là SCM model, điều chỉnh số bước suy luận thành 2
        if is_scm_model and num_inference_steps != 2:
            print(f"WARNING: SCM model detected. Forcing num_inference_steps=2 (was {num_inference_steps})")
            num_inference_steps = 2
            kwargs["num_inference_steps"] = 2
        
        # Thử các phương pháp khác nhau để sinh ảnh
        for attempt, method_name in enumerate([
            "Standard parameters", 
            "Reduced parameters", 
            "Minimal parameters",
            "Original transformer"
        ]):
            try:
                if attempt == 0:
                    # Phương pháp 1: Tham số đầy đủ
                    image = self.pipeline(**kwargs).images[0]
                
                elif attempt == 1:
                    # Phương pháp 2: Giảm tham số
                    print(f"Attempt {attempt+1}: Trying with fewer parameters...")
                    simple_kwargs = {
                        "prompt": prompt,
                        "num_inference_steps": num_inference_steps,  # Already adjusted for SCM if needed
                        "guidance_scale": guidance_scale
                    }
                    image = self.pipeline(**simple_kwargs).images[0]
                
                elif attempt == 2:
                    # Phương pháp 3: Tham số tối thiểu
                    print(f"Attempt {attempt+1}: Trying with minimal parameters...")
                    image = self.pipeline(prompt=prompt).images[0]
                
                else:
                    # Phương pháp 4: Sử dụng transformer gốc
                    print(f"Attempt {attempt+1}: Using original transformer...")
                    if hasattr(self, 'original_transformer') and hasattr(self.pipeline, 'transformer'):
                        # Lưu transformer hiện tại và khôi phục transformer gốc
                        temp_transformer = self.pipeline.transformer
                        self.pipeline.transformer = self.original_transformer
                        
                        try:
                            # Thử sinh ảnh
                            image = self.pipeline(prompt=prompt).images[0]
                        finally:
                            # Khôi phục lại transformer có KV cache
                            self.pipeline.transformer = temp_transformer
                    else:
                        raise RuntimeError("Cannot restore original transformer, no reference available")
                
                # Nếu thành công, thoát khỏi vòng lặp
                print(f"Successfully generated image using method: {method_name}")
                break
                
            except Exception as e:
                print(f"Error with {method_name}: {str(e)}")
                
                # Nếu đã thử tất cả phương pháp
                if attempt == 3:
                    # In thông tin debug
                    print("\n=== Debug Information ===")
                    print(f"Pipeline type: {self.pipeline_type}")
                    print(f"Available components: {list(self.pipeline.components.keys()) if hasattr(self.pipeline, 'components') else 'N/A'}")
                    print(f"KV cached transformer structure: {str(self.kv_cached_transformer.transformer_structure)[:500]}...")
                    
                    raise RuntimeError(f"Failed to generate image after all attempts: {e}")
                    
        # Khôi phục trạng thái ban đầu của use_kv_cache nếu đã thay đổi
        if hasattr(self.kv_cached_transformer, "use_kv_cache") and 'prev_use_cache' in locals():
            self.kv_cached_transformer.use_kv_cache = prev_use_cache
            
        # Tính thời gian
        generation_time = time.time() - start_time
        
        print(f"Image generated in {generation_time:.2f} seconds with KV caching {'enabled' if use_cache else 'disabled'}")
        
        return image, generation_time
    
    def generate_images_with_coco(self, image_filename_to_caption, output_dir, num_images=10, 
                                 num_inference_steps=30, guidance_scale=7.5, use_cache=True):
        """
        Generate images using captions from COCO dataset
        
        Args:
            image_filename_to_caption: Dictionary mapping filenames to captions
            output_dir: Directory to save generated images
            num_images: Number of images to generate
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            use_cache: Whether to use KV cache
            
        Returns:
            generation_times: Dictionary mapping filenames to generation times
            generation_metadata: List of metadata for each generated image
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Limit the number of images to generate
        filenames_captions = list(image_filename_to_caption.items())[:num_images]
        
        generation_times = {}
        generation_metadata = []
        
        # Generate images for each caption
        for i, (filename, prompt) in enumerate(tqdm(filenames_captions, desc="Generating images with COCO captions")):
            output_path = os.path.join(output_dir, filename)
            
            print(f"\n\n{'='*80}")
            print(f"Processing image {i+1}/{len(filenames_captions)}: {filename}")
            print(f"Prompt: {prompt}")
            print(f"Output will be saved to: {output_path}")
            print(f"{'='*80}")
            
            # Skip if the image already exists
            if os.path.exists(output_path):
                print(f"Skipping generation for {filename} (already exists)")
                continue
                
            try:
                # Generate image with KV caching
                image, generation_time = self.generate_image(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    use_cache=use_cache
                )
                
                # Save the image
                image.save(output_path)
                
                print(f"Generated image {i+1}/{len(filenames_captions)}: {filename}")
                print(f"Generation time: {generation_time:.2f} seconds")
                
                # Create metadata
                metadata = {
                    "generated_image_path": output_path,
                    "original_filename": filename,
                    "caption_used": prompt,
                    "generation_time": generation_time,
                    "guidance_scale": guidance_scale,
                    "num_steps": num_inference_steps,
                    "use_cache": use_cache
                }
                
                generation_metadata.append(metadata)
                generation_times[filename] = generation_time
                
            except Exception as e:
                print(f"Error when generating image for prompt '{prompt[:50]}...': {e}")
                generation_times[filename] = -1  # Indicate an error
                
            # Force GC to free memory
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        return generation_times, generation_metadata
    
    def get_vram_usage(self):
        """
        Get current VRAM usage statistics
        
        Returns:
            Dictionary containing VRAM usage information:
            - total_vram_gb: Total VRAM on the GPU in GB
            - free_vram_gb: Free VRAM on the GPU in GB
            - used_vram_gb: Used VRAM on the GPU in GB
            - device_name: Name of the GPU device
        """
        try:
            # First try with pynvml if it's available
            try:
                from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName, nvmlShutdown
                
                nvmlInit()
                device_index = 0  # Use first GPU by default
                handle = nvmlDeviceGetHandleByIndex(device_index)
                memory_info = nvmlDeviceGetMemoryInfo(handle)
                device_name = nvmlDeviceGetName(handle).decode('utf-8')
                
                total_vram_gb = memory_info.total / (1024**3)
                free_vram_gb = memory_info.free / (1024**3)
                used_vram_gb = memory_info.used / (1024**3)
                
                nvmlShutdown()
            except ImportError:
                # If pynvml not available, try nvidia-smi via subprocess
                import subprocess
                
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used,name', '--format=csv,nounits,noheader'],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                values = result.stdout.strip().split(',')
                total_vram_gb = float(values[0]) / 1024
                free_vram_gb = float(values[1]) / 1024
                used_vram_gb = float(values[2]) / 1024
                device_name = values[3].strip()
            
            return {
                "total_vram_gb": total_vram_gb,
                "free_vram_gb": free_vram_gb,
                "used_vram_gb": used_vram_gb,
                "device_name": device_name
            }
        except ImportError:
            # Fallback to torch.cuda
            try:
                if torch.cuda.is_available():
                    device_index = 0
                    device_name = torch.cuda.get_device_name(device_index)
                    total_vram_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
                    free_vram_gb = torch.cuda.memory_reserved(device_index) / (1024**3)
                    used_vram_gb = total_vram_gb - free_vram_gb
                    
                    return {
                        "total_vram_gb": total_vram_gb,
                        "free_vram_gb": free_vram_gb,
                        "used_vram_gb": used_vram_gb,
                        "device_name": device_name
                    }
                else:
                    return {"error": "CUDA not available"}
            except Exception as e:
                return {"error": f"Error getting VRAM info: {str(e)}"}
        except Exception as e:
            return {"error": f"Error getting VRAM info: {str(e)}"}
    
    def monitor_vram_detailed(self, duration=5, interval=0.1, device_index=0):
        """
        Monitor VRAM usage over a specified duration
        
        Args:
            duration: Duration to monitor in seconds
            interval: Sampling interval in seconds
            device_index: GPU device index
            
        Returns:
            Dictionary with VRAM usage statistics
        """
        import time
        
        # Reset VRAM samples and stop event for this run
        global vram_samples, stop_monitoring
        vram_samples.clear()
        stop_monitoring.clear()

        # Start the monitoring thread
        monitor_thread = threading.Thread(target=monitor_vram, args=(device_index,))
        monitor_thread.start()
        
        # Wait for the specified duration
        time.sleep(duration)
        
        # Stop the monitoring thread
        stop_monitoring.set()
        monitor_thread.join()
        
        # Calculate statistics
        if vram_samples:
            avg_vram = sum(vram_samples) / len(vram_samples)
            peak_vram = max(vram_samples)
            min_vram = min(vram_samples)
            
            # Calculate standard deviation
            variance = sum((x - avg_vram) ** 2 for x in vram_samples) / len(vram_samples)
            std_dev = variance ** 0.5
            
            return {
                "average_vram_gb": avg_vram,
                "peak_vram_gb": peak_vram,
                "min_vram_gb": min_vram,
                "std_dev_gb": std_dev,
                "samples_count": len(vram_samples),
                "samples": vram_samples
            }
        else:
            return {"error": "No VRAM samples collected"}

    def benchmark(self, prompt, negative_prompt="", num_inference_steps=30, guidance_scale=7.5, 
                 num_runs=5, use_cache=True, height=1024, width=1024):
        """Benchmark image generation with and without KV caching"""
        results = {
            "with_cache": [],
            "without_cache": [],
            "vram_usage": []
        }
        
        # Chạy với KV caching
        print(f"Benchmarking with KV caching {'enabled' if use_cache else 'disabled'}...")
        for i in range(num_runs):
            try:
                # Get initial VRAM usage
                initial_vram = self.get_vram_usage()
                
                # Truyền tất cả tham số qua generate_image
                _, generation_time = self.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=i,  # Sử dụng index làm seed để tái tạo được kết quả
                    use_cache=use_cache,
                    height=height,
                    width=width
                )
                
                # Get final VRAM usage
                final_vram = self.get_vram_usage()
                
                results["with_cache" if use_cache else "without_cache"].append(generation_time)
                results["vram_usage"].append({
                    "run": i,
                    "cache": use_cache,
                    "initial_vram_gb": initial_vram.get("used_vram_gb"),
                    "final_vram_gb": final_vram.get("used_vram_gb"),
                    "increase_gb": final_vram.get("used_vram_gb", 0) - initial_vram.get("used_vram_gb", 0)
                })
                
            except Exception as e:
                # 4. Sửa lỗi chung - Xử lý ngoại lệ trong benchmark
                print(f"Error during benchmark run {i} with cache={use_cache}: {e}")
                continue
            
            # Giải phóng bộ nhớ
            torch.cuda.empty_cache()
            gc.collect()
        
        # Chạy không có KV caching (nếu trước đó đã chạy với cache)
        if use_cache:
            print("Benchmarking with KV caching disabled...")
            for i in range(num_runs):
                try:
                    # Get initial VRAM usage
                    initial_vram = self.get_vram_usage()
                    
                    # Truyền tất cả tham số qua generate_image
                    _, generation_time = self.generate_image(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed=i,  # Sử dụng index làm seed để tái tạo được kết quả
                        use_cache=False,
                        height=height,
                        width=width
                    )
                    
                    # Get final VRAM usage
                    final_vram = self.get_vram_usage()
                    
                    results["without_cache"].append(generation_time)
                    results["vram_usage"].append({
                        "run": i,
                        "cache": False,
                        "initial_vram_gb": initial_vram.get("used_vram_gb"),
                        "final_vram_gb": final_vram.get("used_vram_gb"),
                        "increase_gb": final_vram.get("used_vram_gb", 0) - initial_vram.get("used_vram_gb", 0)
                    })
                    
                except Exception as e:
                    # 4. Sửa lỗi chung - Xử lý ngoại lệ trong benchmark
                    print(f"Error during benchmark run {i} with cache=False: {e}")
                    continue
                
                # Giải phóng bộ nhớ
                torch.cuda.empty_cache()
                gc.collect()
        
        # Tính toán thống kê
        with_cache_avg = sum(results["with_cache"]) / len(results["with_cache"]) if results["with_cache"] else 0
        without_cache_avg = sum(results["without_cache"]) / len(results["without_cache"]) if results["without_cache"] else 0
        
        speedup = without_cache_avg / with_cache_avg if with_cache_avg > 0 else 0
        
        # Calculate VRAM statistics
        vram_with_cache = [item for item in results["vram_usage"] if item["cache"]]
        vram_without_cache = [item for item in results["vram_usage"] if not item["cache"]]
        
        avg_vram_with_cache = sum(item["increase_gb"] for item in vram_with_cache) / len(vram_with_cache) if vram_with_cache else 0
        avg_vram_without_cache = sum(item["increase_gb"] for item in vram_without_cache) / len(vram_without_cache) if vram_without_cache else 0
        
        print(f"\n===== Benchmark Results =====")
        print(f"Average time with KV caching: {with_cache_avg:.2f} seconds")
        print(f"Average VRAM increase with KV caching: {avg_vram_with_cache:.2f} GB")
        
        if without_cache_avg > 0:
            print(f"Average time without KV caching: {without_cache_avg:.2f} seconds")
            print(f"Average VRAM increase without KV caching: {avg_vram_without_cache:.2f} GB")
            print(f"Speedup factor: {speedup:.2f}x")
            print(f"VRAM efficiency: {(avg_vram_without_cache/avg_vram_with_cache):.2f}x") if avg_vram_with_cache > 0 else None
        
        # Save benchmark results to a file
        timestamp = int(time.time())
        benchmark_file = f"benchmark_results_{timestamp}.json"
        with open(benchmark_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Benchmark results saved to {benchmark_file}")
        
        return results
    
    def interactive_session(self, initial_prompt, num_inference_steps=30, guidance_scale=7.5,
                          negative_prompt="", height=1024, width=1024):
        """Run an interactive session with the same image and different prompts"""
        print("Starting interactive session...")
        print("Generating initial image...")
        
        # Sinh ảnh ban đầu với KV cache bật
        try:
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
            
            # Lưu ảnh ban đầu
            image_filename = f"initial_image_{int(time.time())}.png"
            image.save(image_filename)
            print(f"Initial image saved as {image_filename}")
        except Exception as e:
            # 4. Sửa lỗi chung - Xử lý ngoại lệ trong quá trình sinh ảnh đầu tiên
            print(f"Error generating initial image: {e}")
            print("Trying with simplified parameters...")
            
            try:
                # Thử lại với tham số đơn giản hơn
                with torch.no_grad():
                    image = self.pipeline(prompt=initial_prompt).images[0]
                    past_generation_time = 0
                
                # Lưu ảnh ban đầu
                image_filename = f"initial_image_{int(time.time())}.png"
                image.save(image_filename)
                print(f"Initial image saved as {image_filename}")
            except:
                print("Failed to generate initial image. Exiting interactive session.")
                return
        
        # Vòng lặp tương tác
        while True:
            updated_prompt = input("\nEnter a new prompt (or 'quit' to exit): ")
            if updated_prompt.lower() == 'quit':
                break
            
            print(f"Generating image for: {updated_prompt}")
            
            try:
                # Sinh ảnh mới với giá trị KV cache giữ nguyên
                with torch.no_grad():
                    # Không xóa cache để tái sử dụng
                    new_image, generation_time = self.generate_image(
                        prompt=updated_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        use_cache=True,
                        height=height,
                        width=width
                    )
                
                # Lưu ảnh mới
                new_image_filename = f"image_{int(time.time())}.png"
                new_image.save(new_image_filename)
                print(f"New image saved as {new_image_filename}")
                print(f"Generation time: {generation_time:.2f} seconds (vs. initial: {past_generation_time:.2f} seconds)")
                
                # Cập nhật thời gian để so sánh với lần sau
                past_generation_time = generation_time
            except Exception as e:
                # 4. Sửa lỗi chung - Xử lý ngoại lệ trong quá trình sinh ảnh mới
                print(f"Error generating image: {e}")
                print("Trying with simplified parameters...")
                
                try:
                    # Thử lại với tham số đơn giản hơn
                    with torch.no_grad():
                        new_image = self.pipeline(prompt=updated_prompt).images[0]
                    
                    # Lưu ảnh mới
                    new_image_filename = f"image_{int(time.time())}.png"
                    new_image.save(new_image_filename)
                    print(f"New image saved as {new_image_filename}")
                except:
                    print("Failed to generate image with this prompt.")


def preprocessing_coco(annotations_dir):
    """
    Load COCO captions and image dimensions from annotations
    
    Args:
        annotations_dir: Directory containing COCO annotations
        
    Returns:
        image_filename_to_caption: Dictionary mapping image filenames to captions
        image_dimensions: Dictionary mapping image filenames to dimensions (width, height)
        image_id_to_dimensions: Dictionary mapping image IDs to (width, height, filename)
    """
    # Path to the captions annotation file
    captions_file = os.path.join(annotations_dir, 'captions_val2017.json')

    # Load the captions data
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)

    # Build dictionary mapping image_id to size
    image_id_to_dimensions = {img['id']: (img['width'], img['height'], img['file_name'])
                            for img in captions_data['images']}

    print(f"Read {len(captions_data['annotations'])} captions from COCO annotation file...")
    # To ensure each original image is processed only once for the main prompt purpose
    processed_image_ids = set()

    image_filename_to_caption = {}
    # Store {filename: (width, height)}
    image_dimensions = {}
    # Create a dictionary to store captions by image ID
    for annotation in captions_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']

        if image_id in image_id_to_dimensions and image_id not in processed_image_ids:
            width, height, original_filename = image_id_to_dimensions[image_id]
            image_filename_to_caption[original_filename] = caption
            image_dimensions[original_filename] = (width, height)
            processed_image_ids.add(image_id)

    print(f"Extracted captions for {len(processed_image_ids)} images.")
    print(f"Created mapping for {len(image_filename_to_caption)} images.")
    print(f"Extracted dimensions for {len(image_dimensions)} images from annotations.")
    return image_filename_to_caption, image_dimensions, image_id_to_dimensions

def load_config():
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as file:
            config = json.load(file)
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def evaluate_model_metrics(generation_output_dir, resized_output_dir, val2017_dir, image_filename_to_caption, image_dimensions, metrics_subset=100):
    """
    Evaluate the generated images using various metrics
    
    Args:
        generation_output_dir: Directory containing generated images
        resized_output_dir: Directory containing resized generated images
        val2017_dir: Directory containing original COCO validation images
        image_filename_to_caption: Dictionary mapping image filenames to captions
        image_dimensions: Dictionary mapping image filenames to dimensions
        metrics_subset: Number of images to use for metrics calculation
        
    Returns:
        metrics_results: Dictionary containing calculated metrics
    """
    metrics_results = {}
    
    # Create directory for resized images (needed for FID and PSNR)
    resized_original_dir = os.path.join(generation_output_dir, "resized_original")
    os.makedirs(resized_original_dir, exist_ok=True)
    
    # Calculate FID score
    try:
        print("\n--- Calculating FID Score ---")
        fid_score = calculate_fid(generation_output_dir, resized_output_dir, val2017_dir)
        metrics_results["fid_score"] = fid_score
    except Exception as e:
        print(f"Error calculating FID: {e}")
    
    # Calculate CLIP Score
    try:
        print("\n--- Calculating CLIP Score ---")
        clip_score = calculate_clip_score(generation_output_dir, image_filename_to_caption)
        metrics_results["clip_score"] = clip_score
    except Exception as e:
        print(f"Error calculating CLIP Score: {e}")
    
    # Calculate ImageReward
    try:
        print("\n--- Calculating ImageReward ---")
        image_reward = compute_image_reward(generation_output_dir, image_filename_to_caption)
        metrics_results["image_reward"] = image_reward
    except Exception as e:
        print(f"Error calculating ImageReward: {e}")
    
    # Calculate LPIPS - we need original images to compare with generated images
    try:
        print("\n--- Calculating LPIPS ---")
        # Get list of generated filenames that have been resized
        generated_filenames = [f for f in os.listdir(resized_output_dir) 
                            if os.path.isfile(os.path.join(resized_output_dir, f)) 
                            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        # Use the metrics_subset parameter to limit the number of images for metrics
        subset_size = min(metrics_subset, len(generated_filenames))
        selected_filenames = generated_filenames[:subset_size]
        
        # Manually resize original COCO images to match generated images for LPIPS calculation
        original_dir = os.path.join(val2017_dir)
        for filename in tqdm(selected_filenames, desc="Resizing original images for LPIPS"):
            original_path = os.path.join(original_dir, filename)
            resized_original_path = os.path.join(resized_original_dir, filename)
            
            if os.path.exists(original_path):
                try:
                    # Get the size of the resized generated image for consistency
                    generated_img = Image.open(os.path.join(resized_output_dir, filename))
                    target_size = generated_img.size
                    
                    # Open and resize the original image
                    original_img = Image.open(original_path)
                    resized_original = original_img.resize(target_size, Image.LANCZOS)
                    resized_original.save(resized_original_path)
                except Exception as e:
                    print(f"Error resizing original image {filename}: {e}")
        
        # Now calculate LPIPS using the resized original and generated images
        lpips_score = calculate_lpips(resized_original_dir, resized_output_dir, selected_filenames)
        metrics_results["lpips"] = lpips_score
    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
    
    # Calculate PSNR
    try:
        print("\n--- Calculating PSNR ---")
        # For COCO dataset, we use resized generated images
        original_dir = val2017_dir
        generated_filenames = [f for f in os.listdir(resized_output_dir) 
                              if os.path.isfile(os.path.join(resized_output_dir, f))
                              and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        # Use the metrics_subset parameter to limit the number of images for metrics
        subset_size = min(metrics_subset, len(generated_filenames))
        psnr_score = calculate_psnr_resized(original_dir, resized_output_dir, generated_filenames[:subset_size])
        metrics_results["psnr"] = psnr_score
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
    
    return metrics_results

def load_model_with_fallbacks(model_path=None):
    """Tải model SANA với nhiều cơ chế dự phòng"""
    print("Initializing SANA model with KV caching...")
    
    # Tìm đường dẫn model nếu chưa được chỉ định
    if model_path is None:
        try:
            config = load_config()
            if config and "models" in config:
                # Thử tìm SANA trong config
                for model_key in ["SANA 1.6B", "SANA", "sana"]:
                    if model_key in config["models"]:
                        model_path = config["models"][model_key]["path"]
                        print(f"Found model path in config: {model_path}")
                        break
            
            # Sử dụng đường dẫn mặc định nếu không tìm thấy trong config
            if model_path is None:
                model_path = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
                print(f"Using default model path: {model_path}")
        except Exception as e:
            print(f"Error loading config: {e}")
            model_path = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
            print(f"Using default model path after error: {model_path}")
    
    # Khởi tạo pipeline
    try:
        pipeline = OptimizedSANAPipeline(model_path=model_path)
        print("Model loaded successfully!")
        return pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Cố gắng thử lại với cấu hình khác
        try:
            print("\nTrying to load with alternative configuration...")
            
            # Tải pipeline thông thường không có KV cache
            print("Loading standard diffusion pipeline...")
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # Thử với float16 thay vì bfloat16
                use_safetensors=True,
            )
            pipeline = pipeline.to("cuda")
            
            # Hiển thị thông tin về pipeline đã tải
            print(f"Loaded pipeline type: {type(pipeline).__name__}")
            if hasattr(pipeline, 'components'):
                print(f"Components: {list(pipeline.components.keys())}")
            
            print("\nKV caching not applied due to initialization error")
            return pipeline
        except Exception as e2:
            print(f"Failed to load with alternative configuration: {e2}")
            raise RuntimeError(f"Could not load SANA model after multiple attempts: {e}")

def monitor_generation_vram(pipeline, prompt, use_cache=True, num_inference_steps=30, guidance_scale=7.5, 
                    height=1024, width=1024, negative_prompt="", seed=None):
    """
    Monitor VRAM usage during image generation
    
    Args:
        pipeline: The OptimizedSANAPipeline instance
        prompt: Text prompt for image generation
        use_cache: Whether to use KV cache
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        height: Image height
        width: Image width
        negative_prompt: Negative prompt
        seed: Random seed
        
    Returns:
        Dictionary with VRAM usage statistics and generation time
    """
    import threading
    import time
    import gc
    import torch
    
    # Set up monitoring
    vram_samples = []
    stop_monitoring = threading.Event()
    
    # Define a monitoring function
    def _monitor_vram(device_index=0):
        try:
            # Try with pynvml first
            try:
                from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
                
                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(device_index)
                
                while not stop_monitoring.is_set():
                    try:
                        info = nvmlDeviceGetMemoryInfo(handle)
                        # Convert bytes to GB
                        used_vram_gb = info.used / 1024**3
                        vram_samples.append(used_vram_gb)
                        time.sleep(0.1)  # Sample every 100 milliseconds
                    except Exception as error:
                        print(f"Error during VRAM monitoring: {error}")
                        break
                nvmlShutdown()
            except ImportError:
                # Fall back to torch.cuda
                while not stop_monitoring.is_set():
                    try:
                        if torch.cuda.is_available():
                            # Get current allocated memory in bytes and convert to GB
                            allocated_gb = torch.cuda.memory_allocated(device_index) / 1024**3
                            vram_samples.append(allocated_gb)
                            time.sleep(0.1)  # Sample every 100 milliseconds
                    except Exception as error:
                        print(f"Error during VRAM monitoring with torch.cuda: {error}")
                        break
        except Exception as e:
            print(f"Error in VRAM monitoring: {e}")
    
    # Clean up memory before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    # Start the monitoring thread
    monitor_thread = threading.Thread(target=_monitor_vram)
    monitor_thread.start()
    
    # Run the image generation
    start_time = time.time()
    try:
        if hasattr(pipeline, 'kv_cached_transformer'):
            # Enable/disable KV caching based on parameter
            if hasattr(pipeline.kv_cached_transformer, "use_kv_cache"):
                prev_use_cache = pipeline.kv_cached_transformer.use_kv_cache
                pipeline.kv_cached_transformer.use_kv_cache = use_cache
        
        # Generate the image
        pipeline.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            use_cache=use_cache,
            height=height,
            width=width
        )
        
        # Restore previous KV cache setting
        if hasattr(pipeline, 'kv_cached_transformer') and hasattr(pipeline.kv_cached_transformer, "use_kv_cache"):
            pipeline.kv_cached_transformer.use_kv_cache = prev_use_cache
            
    except Exception as e:
        print(f"Error during image generation: {e}")
    finally:
        generation_time = time.time() - start_time
        
        # Stop the monitoring thread
        stop_monitoring.set()
        monitor_thread.join()
    
    # Calculate statistics
    results = {
        "generation_time_seconds": generation_time,
        "used_kv_cache": use_cache,
    }
    
    if vram_samples:
        avg_vram = sum(vram_samples) / len(vram_samples)
        peak_vram = max(vram_samples)
        min_vram = min(vram_samples)
        initial_vram = vram_samples[0] if vram_samples else 0
        
        # Calculate standard deviation
        variance = sum((x - avg_vram) ** 2 for x in vram_samples) / len(vram_samples)
        std_dev = variance ** 0.5
        
        results.update({
            "average_vram_gb": avg_vram,
            "peak_vram_gb": peak_vram,
            "min_vram_gb": min_vram,
            "initial_vram_gb": initial_vram,
            "vram_increase_gb": peak_vram - initial_vram,
            "std_dev_gb": std_dev,
            "samples_count": len(vram_samples),
        })
        
        # Add a subset of samples for visualization
        if len(vram_samples) > 100:
            # Take every Nth sample to get around 100 samples
            n = max(1, len(vram_samples) // 100)
            results["vram_samples"] = vram_samples[::n]
        else:
            results["vram_samples"] = vram_samples
    else:
        results["error"] = "No VRAM samples collected"
    
    return results

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
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with extra logging")
    # Add COCO-specific arguments
    parser.add_argument("--use_coco", action="store_true",
                        help="Use COCO dataset captions for image generation")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of COCO images to process when --use_coco is enabled")
    parser.add_argument("--skip_metrics", action="store_true",
                        help="Skip calculation of image quality metrics")
    parser.add_argument("--metrics_subset", type=int, default=20,
                        help="Number of images to use for metrics calculation")
    parser.add_argument("--monitor_vram", action="store_true",
                        help="Monitor VRAM usage during generation")
    
    args = parser.parse_args()
    
    # Login to Hugging Face if token provided
    login(token="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")
    
    # Bật chế độ debug nếu được yêu cầu
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print("Debug mode enabled with verbose logging")
    
    try:
        # Tải model với cơ chế dự phòng
        pipeline = load_model_with_fallbacks(model_path=args.model_path)
        
        # Xác định chế độ chạy
        if args.use_coco:
            # Using COCO dataset for image generation
            print("\n=== Running with COCO dataset ===")
            
            # Define paths
            coco_dir = "coco"
            annotations_dir = os.path.join(coco_dir, "annotations")
            val2017_dir = os.path.join(coco_dir, "val2017")
            
            # Check if COCO dataset is available
            if not os.path.exists(annotations_dir) or not os.path.exists(val2017_dir):
                print("COCO dataset not found. Please download it first.")
                return
            
            # Process COCO dataset to get captions
            print("\n=== Loading COCO Captions ===")
            image_filename_to_caption, image_dimensions, image_id_to_dimensions = preprocessing_coco(annotations_dir)
            
            # Create output directories
            generation_output_dir = "kvcache/outputs/coco"
            os.makedirs(generation_output_dir, exist_ok=True)
            print(f"Created directory for generated images: {generation_output_dir}")
            
            resized_output_dir = os.path.join(generation_output_dir, "resized")
            os.makedirs(resized_output_dir, exist_ok=True)
            print(f"Created directory for resized generated images: {resized_output_dir}")
            
            # Create a list to store generation metadata
            generation_metadata = []
            
            # Limit the number of images to generate
            num_images_to_generate = min(args.num_images, len(image_filename_to_caption))
            print(f"\n=== Will generate {num_images_to_generate} images ===")
            
            # Keep track of generation time
            generation_times = {}
            
            # Generate images for COCO captions
            for i, (filename, prompt) in enumerate(tqdm(list(image_filename_to_caption.items())[:num_images_to_generate], desc="Generating images")):
                output_path = os.path.join(generation_output_dir, filename)
                
                print(f"\n\n{'='*80}")
                print(f"Processing image {i+1}/{num_images_to_generate}: {filename} (Prompt: {prompt[:50]}...)")
                print(f"Output will be saved to: {output_path}")
                print(f"{'='*80}")
                
                # Skip if the image already exists
                if os.path.exists(output_path):
                    print(f"Skipping generation for {filename} (already exists)")
                    continue
                    
                try:
                    # Generate image and monitor VRAM
                    # For KV-cached model, we need to use the specialized API
                    if hasattr(pipeline, 'generate_image'):
                        image, generation_time = pipeline.generate_image(
                            prompt=prompt,
                            num_inference_steps=args.steps,
                            guidance_scale=args.guidance_scale,
                            use_cache=True
                        )
                        image.save(output_path)
                        
                        # Create metadata
                        metadata = {
                            "generated_image_path": output_path,
                            "original_filename": filename,
                            "caption_used": prompt,
                            "generation_time": generation_time,
                            "guidance_scale": args.guidance_scale,
                            "num_steps": args.steps
                        }
                        
                        generation_metadata.append(metadata)
                        generation_times[filename] = generation_time
                    else:
                        # Use the resources monitor for standard pipeline
                        generation_time, metadata = pipeline.generate_image(
                            prompt=prompt,
                            num_inference_steps=args.steps,
                            guidance_scale=args.guidance_scale,
                            use_cache=True
                        )
                        generation_metadata.append(metadata)
                        generation_times[filename] = generation_time
                    
                    print(f"Generated image {i+1}/{num_images_to_generate}: {filename}")
                    print(f"Image available at: {output_path}")
                    print(f"{'='*80}\n")
                except Exception as e:
                    print(f"Error when generating image for prompt '{prompt[:50]}...': {e}")
                    generation_times[filename] = -1  # Indicate an error
                    
                # Force GC to free memory
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            print("Image generation complete.")
            
            # Save generation metadata
            metadata_file = os.path.join(generation_output_dir, "sana_kvcache_metadata.json")
            # write_generation_metadata_to_file(metadata_file)
            try:
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(generation_metadata, f, ensure_ascii=False, indent=4)
                print(f"Stored info to metadata: {metadata_file}")
            except Exception as e:
                print(f"Error when saving file metadata JSON: {e}")
            print(f"Stored generation metadata to: {metadata_file}")
            
            # Calculate and print average generation time
            successful_generations = [t for t in generation_times.values() if t > 0]
            print(f"\nGenerated {len(successful_generations)}/{num_images_to_generate} images successfully.")
            if successful_generations:
                avg_time = sum(successful_generations) / len(successful_generations)
                print(f"Average generation time per image: {avg_time:.2f} seconds")
            
            # Resize images for comparison
            try:
                print("\n=== Resizing Images ===")
                resize_images(generation_output_dir, resized_output_dir, image_dimensions)
                print("Image resizing complete.")
            except Exception as e:
                print(f"Error resizing images: {e}")
            
            # Calculate metrics if not skipped
            if not args.skip_metrics:
                print("\n=== Calculating Image Quality Metrics ===")
                metrics_results = evaluate_model_metrics(
                    generation_output_dir,
                    resized_output_dir,
                    val2017_dir,
                    image_filename_to_caption,
                    image_dimensions,
                    args.metrics_subset
                )
                
            # Check if VRAM monitoring is requested
            if args.monitor_vram:
                print("\n=== Running with VRAM monitoring ===")
                # Run with VRAM monitoring
                vram_stats = monitor_generation_vram(
                    pipeline=pipeline,
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    height=args.height,
                    width=args.width,
                    use_cache=True  # Default to using KV cache
                )
                
                # Print VRAM statistics
                print("\n=== VRAM Usage Statistics ===")
                print(f"Generation time: {vram_stats['generation_time_seconds']:.2f} seconds")
                print(f"Average VRAM usage: {vram_stats.get('average_vram_gb', 'N/A'):.2f} GB")
                print(f"Peak VRAM usage: {vram_stats.get('peak_vram_gb', 'N/A'):.2f} GB")
                print(f"Initial VRAM usage: {vram_stats.get('initial_vram_gb', 'N/A'):.2f} GB")
                print(f"VRAM increase: {vram_stats.get('vram_increase_gb', 'N/A'):.2f} GB")
                
                # Save VRAM statistics to a file
                # vram_stats_file = f"vram_stats_{int(time.time())}.json"
                # try:
                #     with open(vram_stats_file, 'w', encoding='utf-8') as f:
                #         json.dump(vram_stats, f, ensure_ascii=False, indent=4)
                #     print(f"VRAM statistics saved to {vram_stats_file}")
                # except Exception as e:
                #     print(f"Error saving VRAM statistics: {e}")
                
                # Print summary of metrics
                print("\n=== Metrics Summary ===")
                for metric, value in metrics_results.items():
                    if value is not None:
                        print(f"{metric}: {value:.4f}")
                    else:
                        print(f"{metric}: N/A")
            
                # Save summary report
                summary_file = os.path.join(generation_output_dir, "sana_kvcache_summary.txt")
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write("=== SANA KV-Cache Generation Summary ===\n\n")
                    f.write(f"Processed {num_images_to_generate} COCO captions\n")
                    f.write(f"Inference steps: {args.steps}\n")
                    f.write(f"Guidance scale: {args.guidance_scale}\n")
                    
                    if successful_generations:
                        f.write(f"\nGeneration Statistics:\n")
                        f.write(f"Successfully generated {len(successful_generations)}/{num_images_to_generate} images\n")
                        f.write(f"Average generation time: {avg_time:.2f} seconds per image\n")
                        f.write(f"Total generation time: {sum(successful_generations):.2f} seconds\n")
                        # Add VRAM usage to summary
                        if args.monitor_vram and vram_stats:
                            f.write(f"\nVRAM Usage Statistics:\n")
                            f.write(f"Average VRAM usage: {vram_stats.get('average_vram_gb', 'N/A'):.2f} GB\n")
                            f.write(f"Peak VRAM usage: {vram_stats.get('peak_vram_gb', 'N/A'):.2f} GB\n")
                            f.write(f"Initial VRAM usage: {vram_stats.get('initial_vram_gb', 'N/A'):.2f} GB\n")
                            f.write(f"VRAM increase: {vram_stats.get('vram_increase_gb', 'N/A'):.2f} GB\n")
                    # Add metrics results to summary
                    if metrics_results:
                        f.write("\n=== Image Quality Metrics ===\n")
                        for metric_name, metric_value in metrics_results.items():
                            if metric_value is not None:
                                f.write(f"{metric_name.upper()}: {metric_value:.4f}\n")
                                
                print(f"\nCompleted COCO image generation. Results saved to {generation_output_dir}")
                print(f"Summary report: {summary_file}")
            else:
                print("\n=== Skipping Image Quality Metrics (--skip_metrics flag set) ===")  
        
        elif args.benchmark:
            # Chạy benchmark
            if hasattr(pipeline, 'benchmark'):
                pipeline.benchmark(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    num_runs=args.num_runs,
                    height=args.height,
                    width=args.width
                )
            else:
                print("Benchmark mode not available for this pipeline")
                
        elif args.interactive:
            # Chế độ tương tác
            if hasattr(pipeline, 'interactive_session'):
                pipeline.interactive_session(
                    initial_prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width
                )
            else:
                print("Interactive mode not available for this pipeline")
                
        else:
            # Sinh một ảnh
            start_time = time.time()
                
            # Kiểm tra nếu đây là pipeline thông thường hay pipeline tùy chỉnh
            if hasattr(pipeline, 'generate_image'):
                image, _ = pipeline.generate_image(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    height=args.height,
                    width=args.width
                )
            else:
                # Sử dụng cách gọi tiêu chuẩn nếu không có generate_image
                generator = None if args.seed is None else torch.Generator("cuda").manual_seed(args.seed)
                
                output = pipeline(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    generator=generator
                )
                
                image = output.images[0]
                print(f"Image generated in {time.time() - start_time:.2f} seconds")
            
            # Lưu ảnh
            output_filename = f"output_{int(time.time())}.png"
            image.save(output_filename)
            print(f"Image saved as {output_filename}")
            
    except Exception as e:
        # 4. Sửa lỗi chung - Hiển thị thông báo lỗi chi tiết
        print(f"\n===== Error running SANA model =====")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
