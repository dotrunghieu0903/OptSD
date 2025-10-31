"""
Flash Attention package initialization for diffusion models

This package implements Flash Attention optimization for diffusion models.
It provides both PyTorch and Triton implementations of the FlashAttention-2 algorithm.
"""

from flash_attn.flash_attn_module import FlashAttentionModule, FlashMHA
from flash_attn.flash_attn_integration import FlashAttentionKVCacheTransformer
from flash_attn.flash_attn_pipeline import FlashAttentionOptimizedPipeline

__all__ = [
    'FlashAttentionModule',
    'FlashMHA',
    'FlashAttentionKVCacheTransformer',
    'FlashAttentionOptimizedPipeline'
]