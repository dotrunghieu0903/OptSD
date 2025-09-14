#!/usr/bin/env python3
"""
Flash Attention Module for Diffusion Models

This module implements Flash Attention for diffusion models based on the FlashAttention-2 algorithm.
It provides both PyTorch and Triton implementations for efficient attention calculation.

References:
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- Original implementation: https://github.com/Dao-AILab/flash-attention
"""

import torch
import math
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

# ======================================================================
# PYTORCH IMPLEMENTATION
# ======================================================================

class FlashAttention2Function(torch.autograd.Function):
    """
    PyTorch implementation of the FlashAttention-2 forward pass.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Get dimensions from input tensors following the (B, H, N, D) convention
        B, H, N_Q, D_H = Q.shape
        _, _, N_K, _ = K.shape

        # Define tile sizes
        Q_TILE_SIZE = 128
        K_TILE_SIZE = 128
        
        N_Q_tiles = math.ceil(N_Q / Q_TILE_SIZE)
        N_K_tiles = math.ceil(N_K / K_TILE_SIZE)

        # Initialize final output tensors
        O_final = torch.zeros_like(Q, dtype=Q.dtype)
        L_final = torch.zeros((B, H, N_Q), device=Q.device, dtype=torch.float32)
        
        scale = 1.0 / math.sqrt(D_H)

        # Main loops: Iterate over each batch and head
        for b in range(B):
            for h in range(H):
                Q_bh = Q[b, h, :, :]
                K_bh = K[b, h, :, :]
                V_bh = V[b, h, :, :]

                # Loop over query tiles
                for i in range(N_Q_tiles):
                    q_start = i * Q_TILE_SIZE
                    q_end = min((i + 1) * Q_TILE_SIZE, N_Q)
                    Q_tile = Q_bh[q_start:q_end, :]

                    # Initialize accumulators for this query tile
                    o_i = torch.zeros_like(Q_tile, dtype=Q.dtype)
                    l_i = torch.zeros(q_end - q_start, device=Q.device, dtype=torch.float32)
                    m_i = torch.full((q_end - q_start,), -float('inf'), device=Q.device, dtype=torch.float32)

                    # Inner loop over key/value tiles
                    for j in range(N_K_tiles):
                        k_start = j * K_TILE_SIZE
                        k_end = min((j + 1) * K_TILE_SIZE, N_K)

                        K_tile = K_bh[k_start:k_end, :]
                        V_tile = V_bh[k_start:k_end, :]
                        
                        S_ij = (Q_tile @ K_tile.transpose(-1, -2)) * scale
                        
                        # Apply causal masking if needed
                        if is_causal:
                            row_ids = torch.arange(q_start, q_end, device=Q.device).unsqueeze(-1)
                            col_ids = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                            causal_mask = row_ids < col_ids
                            S_ij = S_ij.masked_fill(causal_mask, -float('inf'))
                        
                        # Compute the new running maximum
                        m_ij, _ = torch.max(S_ij, dim=-1)
                        m_new = torch.maximum(m_i, m_ij)
                        
                        # Rescale the previous accumulators with proper dtype
                        o_i = o_i * torch.exp(m_i - m_new).unsqueeze(-1)
                        l_i = l_i * torch.exp(m_i - m_new)
                        
                        # Compute the probabilities for the current tile
                        P_tilde_ij = torch.exp(S_ij - m_new.unsqueeze(-1)).to(Q.dtype)
                        
                        # Accumulate the current tile's contribution
                        l_i = l_i + torch.sum(P_tilde_ij, dim=-1)
                        o_i = o_i + P_tilde_ij @ V_tile
                        
                        # Update the running max for the next iteration
                        m_i = m_new

                    # After iterating through all key tiles, normalize the output
                    l_i_reciprocal = torch.where(l_i > 0, 1.0 / l_i, 0)
                    o_i_normalized = o_i * l_i_reciprocal.unsqueeze(-1)
                    
                    L_tile = m_i + torch.log(l_i)
                    
                    # Write results for this tile back to the final output tensors
                    O_final[b, h, q_start:q_end, :] = o_i_normalized
                    L_final[b, h, q_start:q_end] = L_tile
        
        O_final = O_final.to(Q.dtype)

        ctx.save_for_backward(Q, K, V, O_final, L_final)
        ctx.is_causal = is_causal
 
        return O_final, L_final
    
    @staticmethod
    def backward(ctx, grad_out, grad_L):
        # Backward pass not implemented for this example
        # In a full implementation, this would compute gradients
        raise NotImplementedError("Backward pass not yet implemented for FlashAttention2Function")

# ======================================================================
# TRITON IMPLEMENTATION
# ======================================================================

@triton.jit
def _flash_attention_forward_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA.
    """
    # Identify the block of queries and the batch/head to be processed
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # Map Query Head to Shared K/V Head for GQA support
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # Initialize accumulators in SRAM
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Load the block of queries (Q_i)
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504

    # Process blocks of keys and values
    # When IS_CAUSAL, we need special handling for diagonal blocks
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        # Load K_j block
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        # Compute attention scores
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        
        # Apply causal masking if needed
        if IS_CAUSAL:
            q_pos = q_offsets[:, None]
            k_pos = k_offsets[None, :]
            causal_mask = q_pos < k_pos  # q_idx < k_idx means we mask it
            s_ij = tl.where(causal_mask, float('-inf'), s_ij)
        
        # Load V_j block
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        exp_scale = tl.exp2((m_i - m_new))
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc = acc + tl.dot(p_ij, v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new

    # Normalize and write the final output block
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)

def flash_attention_forward_triton(q, k, v, is_causal=False):
    """
    Flash attention implementation using Triton.
    
    Args:
        q: Query tensor of shape (batch_size, n_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, n_kv_heads, seq_len, head_dim)
        v: Value tensor of shape (batch_size, n_kv_heads, seq_len, head_dim)
        is_causal: Whether to apply causal masking
        
    Returns:
        Output tensor of shape (batch_size, n_heads, seq_len, head_dim)
    """
    # Shape checks
    batch_size, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    # Verify dimensions
    assert q.shape[0] == k.shape[0] == v.shape[0]
    assert k.shape == v.shape
    assert head_dim <= 128, "Head dimension must be <= 128"
    assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of key/value heads"
    
    # Store original dtype and ensure we're using a supported type
    orig_dtype = q.dtype
    if orig_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
        raise ValueError(f"Unsupported dtype: {orig_dtype}. Must be float16, bfloat16, or float32.")
    
    # Output tensor
    output = torch.empty_like(q)
    
    # Softmax scaling factor
    scale = 1.0 / math.sqrt(head_dim)
    
    # Define block sizes
    BLOCK_M = 128  # query block size
    BLOCK_N = 128  # key/value block size
    
    # Compute strides
    q_strides = (q.stride(0), q.stride(1), q.stride(2))
    k_strides = (k.stride(0), k.stride(1), k.stride(2))
    v_strides = (v.stride(0), v.stride(1), v.stride(2))
    
    # Launch the kernel
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * n_q_heads)
    
    _flash_attention_forward_kernel[grid](
        q, k, v, output,
        *q_strides, *k_strides, *v_strides,
        scale, seq_len, n_q_heads, n_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
    )
    
    return output.to(orig_dtype)

# ======================================================================
# MODULE WRAPPERS
# ======================================================================

class FlashAttentionModule(torch.nn.Module):
    """
    FlashAttention module that can be used as a drop-in replacement for attention layers.
    
    This module supports both PyTorch and Triton implementations of FlashAttention.
    """
    
    def __init__(self, use_triton=True):
        super().__init__()
        self.use_triton = use_triton and torch.cuda.is_available() and triton.is_available()
    
    def forward(self, q, k, v, is_causal=False):
        """
        Forward pass using FlashAttention
        
        Args:
            q: Query tensor of shape (batch_size, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, n_kv_heads, seq_len, head_dim)
            v: Value tensor of shape (batch_size, n_kv_heads, seq_len, head_dim)
            is_causal: Whether to apply causal masking
            
        Returns:
            Output tensor of shape (batch_size, n_heads, seq_len, head_dim)
        """
        if self.use_triton:
            try:
                return flash_attention_forward_triton(q, k, v, is_causal)
            except Exception as e:
                print(f"Triton implementation failed with error: {e}. Falling back to PyTorch implementation.")
                output, _ = FlashAttention2Function.apply(q, k, v, is_causal)
                return output
        else:
            output, _ = FlashAttention2Function.apply(q, k, v, is_causal)
            return output

class FlashMHA(torch.nn.Module):
    """
    Multi-head attention module using FlashAttention
    
    This module handles projections of inputs to query, key, and value tensors,
    applies flash attention, and projects the output.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        causal: bool = False,
        use_triton: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.use_triton = use_triton
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        
        self.flash_attn = FlashAttentionModule(use_triton=use_triton)
        
        self.batch_first = batch_first
        
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention with FlashAttention
        """
        # Default behavior: if key/value not provided, use query for both
        if key is None:
            key = query
        if value is None:
            value = key
        
        # Handle batch dimension
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # Project inputs to q, k, v
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        batch_size, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape
        
        # Reshape: [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        # Transpose: [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply flash attention
        attn_output = self.flash_attn(q, k, v, is_causal=self.causal)
        
        # Reshape: [batch_size, num_heads, seq_len_q, head_dim] -> [batch_size, seq_len_q, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        
        # Project outputs
        output = self.out_proj(attn_output)
        
        # Handle batch dimension for output
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        # Return without attention weights since we don't compute them
        return output, None