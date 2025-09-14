# Problem 6+7: Extending for Sliding Window Attention(SWA) and Sink Attention(StreamingLLM)
# Extend the Triton kernel from Problem 5 to support Sliding Window Attention and Sink Attention.
# In Sliding Window Attention, each query position attends to a fixed-size window of key positions around it.
# In Sink Attention, certain "sink" positions can attend to all previous key positions.
# Reference: https://triton-lang.org/
# Problem 6
import torch
import math

import triton
import triton.language as tl

@triton.jit
def _flash_attention_forward_swa_kernel(
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
    WINDOW_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel template for Sliding Window Attention (SWA) with GQA.
    """
    # 1. Boilerplate setup
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- STUDENT IMPLEMENTATION REQUIRED (Part 1: GQA Logic) ---
    # Map from query heads to key/value heads for GQA
    num_heads_per_group = N_Q_HEADS // N_KV_HEADS  # Number of query heads per group
    kv_head_idx = q_head_idx // num_heads_per_group  # Integer division for correct mapping
    # --- END OF GQA IMPLEMENTATION ---


    # 2. Initialize accumulators
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load query block
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED (Part 2: SWA Logic) ---
    # Implement sliding window attention by limiting context to WINDOW_SIZE
    # Calculate the starting position based on current block position and window size
    # Max between 0 and (current block start - window size)
    window_start = tl.maximum(0, q_block_idx * BLOCK_M - WINDOW_SIZE)
    window_start = (window_start // BLOCK_N) * BLOCK_N

    # --- Phase 1: Off-Diagonal Blocks (within the window) ---
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
        # Load K_j block
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        # Compute attention scores
        s_ij = tl.dot(q_block, k_block)
        s_ij = s_ij * qk_scale
        
        # Apply sliding window mask
        q_pos = q_offsets[:, None]
        k_pos = k_offsets[None, :]
        window_mask = (q_pos - k_pos) >= WINDOW_SIZE
        s_ij = tl.where(window_mask, float('-inf'), s_ij)
        
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

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
        # Load K_j block with causal masking for diagonal blocks
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        # Compute attention scores
        s_ij = tl.dot(q_block, k_block)
        s_ij = s_ij * qk_scale
        
        # Apply both causal and sliding window masks
        q_pos = q_offsets[:, None]
        k_pos = k_offsets[None, :]
        causal_mask = q_pos >= k_pos  # Causal masking
        window_mask = (q_pos - k_pos) >= WINDOW_SIZE  # Sliding window masking
        mask = causal_mask | window_mask
        s_ij = tl.where(mask, float('-inf'), s_ij)
        
        # Load V_j block
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Online softmax update with masked attention scores
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        exp_scale = tl.exp2((m_i - m_new))
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale

        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc = acc + tl.dot(p_ij, v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new
    # --- END OF SWA IMPLEMENTATION ---


    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)

def flash_attention_forward(q, k, v, is_causal=True, window_size=128):
    
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel.
    """
    batch, n_q_heads, seq_len, head_dim = q.shape
    n_kv_heads = k.shape[1]
    
    assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"
    assert is_causal, "This kernel is only supported for causal attention"

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
        
        # Apply sliding window mask
        idx = torch.arange(seq_len, device=s.device)
        row = idx.unsqueeze(1)
        col = idx.unsqueeze(0)
        sliding_mask = (col <= row) & (col >= row - (window_size - 1))
        s = s.masked_fill(~sliding_mask, float('-inf'))
        
        # Apply softmax and compute output
        p = torch.softmax(s, dim=-1)
        o = p @ v_expanded.float()
        return o.to(orig_dtype)
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    # if window_size != 4096:
    #     raise ValueError("This kernel is compiled for a fixed window size of 4096")

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        WINDOW_SIZE=window_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o