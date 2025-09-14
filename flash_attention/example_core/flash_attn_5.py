# Problem 5: Extending for Grouped-Query Attention (GQA)
# Extend the Triton kernel from Problem 4 to support Grouped-Query Attention (GQA).
# In GQA, the number of query heads (H_q) can be different from the number of key/value heads (H_kv).
# Each query head attends to a specific group of key/value heads.
# Reference: https://triton-lang.org/

import torch
import math

import triton
import triton.language as tl

@triton.jit
def _flash_attention_forward_gqa_kernel(
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
):
    """
    Triton kernel template for the forward pass of causal FlashAttention with GQA.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 1) ---
    # Map current query head to corresponding shared key/value head
    num_heads_per_group = N_Q_HEADS // N_KV_HEADS  # Number of query heads per group
    kv_head_idx = q_head_idx // num_heads_per_group  # Integer division for KV head mapping
    # --- END OF STUDENT IMPLEMENTATION ---


    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504
    
    # --- Phase 1: Off-Diagonal Blocks ---
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 2) ---
        # Load K_j block using kv_head_idx instead of q_head_idx
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        # Compute attention scores
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        
        # Load V_j block using kv_head_idx
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Online softmax update (reused from Problem 4)
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        exp_scale = tl.exp2((m_i - m_new))
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        # Compute attention probabilities and update accumulators
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc = acc + tl.dot(p_ij, v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new
        # --- END OF STUDENT IMPLEMENTATION ---

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 3) ---
        # Load K_j block for diagonal block using kv_head_idx
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        # Compute attention scores
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        
        # Apply causal mask: row_idx >= col_idx for diagonal block
        q_pos = q_offsets[:, None]
        k_pos = k_offsets[None, :]
        causal_mask = q_pos < k_pos
        s_ij = tl.where(causal_mask, float('-inf'), s_ij)
        
        # Load V_j block using kv_head_idx
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Online softmax update with causal masking (reused from Problem 4)
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        exp_scale = tl.exp2((m_i - m_new))
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        # Compute attention probabilities and update accumulators
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc = acc + tl.dot(p_ij, v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new
        # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = l_i[:, None] + 1e-6
    acc = acc / l_i_safe
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)

def flash_attention_forward(q, k, v, is_causal=True):
    """
    Python wrapper for the GQA-enabled causal FlashAttention kernel.
    """
    batch, n_q_heads, seq_len, head_dim = q.shape
    n_kv_heads = k.shape[1]
    
    assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"
    
    if q.dtype != torch.float32:
        orig_dtype = q.dtype
        num_groups = n_q_heads // n_kv_heads
        scale = 1.0 / math.sqrt(head_dim)
        
        # Expand K and V for each group
        k_expanded = k.unsqueeze(2).expand(batch, n_kv_heads, num_groups, seq_len, head_dim)
        k_expanded = k_expanded.reshape(batch, n_q_heads, seq_len, head_dim)
        v_expanded = v.unsqueeze(2).expand(batch, n_kv_heads, num_groups, seq_len, head_dim)
        v_expanded = v_expanded.reshape(batch, n_q_heads, seq_len, head_dim)
        
        # Compute attention scores
        s = (q.float() @ k_expanded.transpose(-1, -2).float()) * scale
        
        if is_causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=s.device, dtype=torch.bool), diagonal=1)
            s = s.masked_fill(mask, float('-inf'))
        
        # Apply softmax and compute output
        p = torch.softmax(s, dim=-1)
        o = p @ v_expanded.float()
        return o.to(orig_dtype)
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_gqa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o