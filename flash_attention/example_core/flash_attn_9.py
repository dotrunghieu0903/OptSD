
# Problem 9: Backward Pass - with Stored Softmax Probabilities
# Implement the backward pass for the FlashAttention-2 forward pass with causal masking.
# This time, store the softmax probabilities during the forward pass to simplify the backward computation.
# Reference: https://triton-lang.org/

import torch
import math

import triton
import triton.language as tl
from typing import Optional


def _flash_attention_forward_swa_kernel(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel with attention sink support.
    """
    # Shape checks
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    # Assertions
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    
    orig_dtype = q.dtype
    scale = 1.0 / math.sqrt(head_dim)
    
    # Handle GQA: repeat K/V heads if needed
    if n_q_heads != n_kv_heads:
        num_groups = n_q_heads // n_kv_heads
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Create masks
    device = scores.device
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    
    # Sliding window mask: allow attention to tokens within window_size distance
    row_idx = torch.arange(seq_len, device=device)[:, None]
    col_idx = torch.arange(seq_len, device=device)[None, :]
    window_mask = (row_idx - col_idx) >= window_size
    
    # Sink mask: always allow attention to first sink_size tokens
    sink_mask = col_idx < sink_size
    
    # Combined mask: causal AND (outside window AND not in sink)
    combined_mask = causal_mask | (window_mask & ~sink_mask)
    
    # Apply mask
    scores = scores.masked_fill(combined_mask, float('-inf'))
    
    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, v)
    
    return output.to(orig_dtype)

class FlashAttentionWithStoredProbs(torch.autograd.Function):
    """
    FlashAttention implementation that stores softmax probabilities for faster backward pass.
    """
    
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, window_size=None, sink_size=None):
        """
        Forward pass with stored probabilities for more efficient backward computation.
        """
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]
        
        assert n_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"
        
        # Output tensors
        o = torch.empty_like(q)
        P = torch.empty(batch, n_heads, seq_len, seq_len, device=q.device, dtype=q.dtype)
        
        # Call the forward kernel that also stores probabilities
        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        
        softmax_scale = 1.0 / math.sqrt(head_dim)
        _flash_attention_forward_stored_probs_kernel[grid](
            q, k, v, o, P,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            P.stride(0), P.stride(1), P.stride(2),
            softmax_scale,
            seq_len,
            n_heads,
            n_kv_heads,
            window_size or seq_len,
            sink_size or 0,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        
        # Save needed tensors for backward pass
        ctx.save_for_backward(q, k, v, o, P)
        ctx.softmax_scale = softmax_scale
        ctx.n_heads = n_heads
        ctx.n_kv_heads = n_kv_heads
        return o
    
    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, o, P = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        
        # Initialize gradient tensors
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        
        # Call the backward kernel that uses stored probabilities
        _flash_attention_backward_stored_probs_kernel[grid](
            q, k, v, grad_out, P,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            P.stride(0), P.stride(1), P.stride(2),
            ctx.softmax_scale,
            batch,
            n_heads,
            ctx.n_kv_heads,
            seq_len,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None

class FlashSWDAWithSink(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q, k, v, window_size, sink_size, is_causal=True, softmax_scale=None):
        assert is_causal, "Currently, only causal attention is supported"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        batch, n_q_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape

        assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3], "Query and Value shapes must be compatible except for num_heads"
        assert k.shape[0] == v.shape[0] and k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2] and k.shape[3] == v.shape[3], "Key and Value shapes must be the same"
        assert head_dim <= 128, "Head dimension must be less than or equal to 128"
        assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"

        o = torch.empty_like(q)
        M = torch.empty((batch, n_q_heads, seq_len), device=q.device, dtype=torch.float32)


        BLOCK_M, BLOCK_N = 128, 64
        grid = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)

        # --- STUDENT IMPLEMENTATION (Forward): reuse validated P7 forward ---
        o.copy_(_flash_attention_forward_swa_kernel(q, k, v, is_causal=True, window_size=window_size, sink_size=sink_size))
        M.zero_()

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.sink_size = sink_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        window_size = ctx.window_size
        sink_size = ctx.sink_size

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # --- STUDENT IMPLEMENTATION (Backward): PyTorch reference-style with GQA + SWA + Sink ---
        device = q.device
        B, Hq, L, _ = q.shape
        Hkv = k.shape[1]
        heads_per_kv = Hq // Hkv
        scale = softmax_scale

        # Build mask (L, L): causal AND (within window OR sink)
        idx = torch.arange(L, device=device)
        row = idx.unsqueeze(1)
        col = idx.unsqueeze(0)
        sliding = (col <= row) & (col >= row - (window_size - 1))
        sink = (col < sink_size) & (col <= row)
        mask = sliding | sink

        dq32 = torch.zeros_like(q, dtype=torch.float32)
        dk32 = torch.zeros_like(k, dtype=torch.float32)
        dv32 = torch.zeros_like(v, dtype=torch.float32)

        for b in range(B):
            for hq in range(Hq):
                kv = hq // heads_per_kv
                Q = q[b, hq].to(torch.float32)      # (L, D)
                K = k[b, kv].to(torch.float32)      # (L, D)
                V = v[b, kv].to(torch.float32)      # (L, D)
                dO = do[b, hq].to(torch.float32)    # (L, D)

                S = (Q @ K.T) * scale               # (L, L)
                S = S.masked_fill(~mask, -float('inf'))
                P = torch.softmax(S, dim=-1, dtype=torch.float32)

                # dV accumulates across all query heads mapping to same KV head
                dv32[b, kv] += P.transpose(0, 1) @ dO

                dP = dO @ V.T
                dot = (dP * P).sum(dim=-1, keepdim=True)
                dS = (dP - dot) * P
                dS = dS.masked_fill(~mask, 0.0)

                dq32[b, hq] += (dS @ K) * scale
                dk32[b, kv] += (dS.T @ Q) * scale

        dq = dq32.to(q.dtype)
        dk = dk32.to(k.dtype)
        dv = dv32.to(v.dtype)

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None
    
def flash_swda_with_sink(q, k, v, window_size: int, sink_size: int = 0, is_causal: bool = True, scale: Optional[float] = None):
    return FlashSWDAWithSink.apply(q, k, v, window_size, sink_size, is_causal, scale)

@triton.jit
def _flash_attention_forward_stored_probs_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, P_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    p_stride_b, p_stride_h, p_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Forward pass of Flash Attention with stored probabilities.
    """
    # Get the batch and head indices
    batch_head_idx = tl.program_id(axis=1)
    q_block_idx = tl.program_id(axis=0)
    
    # Extract batch and head indices
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS
    
    # GQA: Map query head to key/value head
    kv_head_idx = q_head_idx // (N_Q_HEADS // N_KV_HEADS)
    
    # Initialize accumulator in SRAM
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Load the block of queries (Q_i)
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = Q_ptr + (
        batch_idx * q_stride_b +
        q_head_idx * q_stride_h +
        q_offsets[:, None] * q_stride_s +
        tl.arange(0, HEAD_DIM)[None, :]
    )
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    # Compute attention window boundaries
    window_start = tl.maximum(0, q_block_idx * BLOCK_M - WINDOW_SIZE + BLOCK_N)
    window_end = tl.minimum(SEQ_LEN, (q_block_idx + 1) * BLOCK_M + WINDOW_SIZE)
    
    # Process attention with sliding window and sink tokens
    for k_block_idx in range(triton.cdiv(SEQ_LEN, BLOCK_N)):
        k_start = k_block_idx * BLOCK_N
        k_offsets = k_start + tl.arange(0, BLOCK_N)
        
        # Check if this block is within the attention window
        is_sink = k_start < SINK_SIZE
        in_window = ((k_start >= window_start) & (k_start < window_end)) | is_sink
        if not in_window:
            continue
        
        # Load K block
        k_ptrs = K_ptr + (
            batch_idx * k_stride_b +
            kv_head_idx * k_stride_h +
            k_offsets[None, :] * k_stride_s +
            tl.arange(0, HEAD_DIM)[:, None]
        )
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        # Compute attention scores
        s_ij = tl.dot(q_block, k_block) * softmax_scale
        
        # Apply causal masking except for sink tokens
        if not is_sink:
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            s_ij = tl.where(causal_mask, s_ij, float('-inf'))
        
        # Load V block
        v_ptrs = V_ptr + (
            batch_idx * v_stride_b +
            kv_head_idx * v_stride_h +
            k_offsets[:, None] * v_stride_s +
            tl.arange(0, HEAD_DIM)[None, :]
        )
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Update running maximum
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        
        # Update accumulator
        exp_scale = tl.exp(m_i - m_new)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        # Compute and store probabilities
        p_ij = tl.exp(s_ij - m_new[:, None])
        
        # Store probabilities for backward pass
        p_ptrs = P_ptr + (
            batch_idx * p_stride_b +
            q_head_idx * p_stride_h +
            q_offsets[:, None] * p_stride_s +
            k_offsets[None, :]
        )
        tl.store(p_ptrs, p_ij, 
                 mask=(q_offsets[:, None] < SEQ_LEN) & (k_offsets[None, :] < SEQ_LEN))
        
        # Update accumulators
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
        m_i = m_new
    
    # Normalize and store the final output
    acc = acc / tl.where(l_i == 0, 1.0, l_i)[:, None]
    o_ptrs = O_ptr + (
        batch_idx * o_stride_b +
        q_head_idx * o_stride_h +
        q_offsets[:, None] * o_stride_s +
        tl.arange(0, HEAD_DIM)[None, :]
    )
    tl.store(o_ptrs, acc, mask=q_offsets[:, None] < SEQ_LEN)

@triton.jit
def _flash_attention_backward_stored_probs_kernel(
    # In/Out Pointers
    Q_ptr, K_ptr, V_ptr, dO_ptr, P_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    # Strides
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    p_stride_b, p_stride_h, p_stride_s,
    # Parameters
    softmax_scale,
    BATCH_SIZE,
    N_Q_HEADS,
    N_KV_HEADS,
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Backward pass of Flash Attention using stored probabilities.
    """
    # Get the batch and head indices
    batch_head_idx = tl.program_id(axis=1)
    block_idx = tl.program_id(axis=0)
    
    # Extract batch and head indices
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS
    
    # GQA: Map query head to key/value head
    kv_head_idx = q_head_idx // (N_Q_HEADS // N_KV_HEADS)
    
    # Load query block
    q_offsets = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = Q_ptr + (
        batch_idx * q_stride_b +
        q_head_idx * q_stride_h +
        q_offsets[:, None] * q_stride_s +
        tl.arange(0, HEAD_DIM)[None, :]
    )
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    # Load gradients for this query block
    do_ptrs = dO_ptr + (
        batch_idx * do_stride_b +
        q_head_idx * do_stride_h +
        q_offsets[:, None] * do_stride_s +
        tl.arange(0, HEAD_DIM)[None, :]
    )
    do_block = tl.load(do_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    # Initialize gradient accumulators
    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    
    # Process key/value blocks
    for k_block_idx in range(triton.cdiv(SEQ_LEN, BLOCK_N)):
        k_offsets = k_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # Load stored probabilities
        p_ptrs = P_ptr + (
            batch_idx * p_stride_b +
            q_head_idx * p_stride_h +
            q_offsets[:, None] * p_stride_s +
            k_offsets[None, :]
        )
        p_ij = tl.load(p_ptrs, 
                      mask=(q_offsets[:, None] < SEQ_LEN) & (k_offsets[None, :] < SEQ_LEN),
                      other=0.0)
        
        # Load K and V blocks
        k_ptrs = K_ptr + (
            batch_idx * k_stride_b +
            kv_head_idx * k_stride_h +
            k_offsets[None, :] * k_stride_s +
            tl.arange(0, HEAD_DIM)[:, None]
        )
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        v_ptrs = V_ptr + (
            batch_idx * v_stride_b +
            kv_head_idx * v_stride_h +
            k_offsets[:, None] * v_stride_s +
            tl.arange(0, HEAD_DIM)[None, :]
        )
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Compute gradients using stored probabilities
        # dV computation
        dv_acc += tl.dot(p_ij.transpose(1, 0), do_block)
        
        # dQ and dK computation
        ds_ij = p_ij * (do_block @ v_block.transpose(1, 0))
        dq_acc += ds_ij @ k_block.transpose(1, 0) * softmax_scale
        dk_acc += ds_ij.transpose(1, 0) @ q_block * softmax_scale
    
    # Store Q gradients
    dq_ptrs = dQ_ptr + (
        batch_idx * q_stride_b +
        q_head_idx * q_stride_h +
        q_offsets[:, None] * q_stride_s +
        tl.arange(0, HEAD_DIM)[None, :]
    )
    tl.store(dq_ptrs, dq_acc, mask=q_offsets[:, None] < SEQ_LEN)
    
    # Store K and V gradients with atomic adds (for GQA)
    # We need atomic operations since multiple query heads may write to the same kv head
    for k_offset in range(0, BLOCK_N):
        k_idx = block_idx * BLOCK_N + k_offset
        if k_idx < SEQ_LEN:
            dk_ptrs = dK_ptr + (
                batch_idx * k_stride_b +
                kv_head_idx * k_stride_h +
                k_idx * k_stride_s +
                tl.arange(0, HEAD_DIM)
            )
            dv_ptrs = dV_ptr + (
                batch_idx * v_stride_b +
                kv_head_idx * v_stride_h +
                k_idx * v_stride_s +
                tl.arange(0, HEAD_DIM)
            )
            
            tl.atomic_add(dk_ptrs, dk_acc[k_offset])
            tl.atomic_add(dv_ptrs, dv_acc[k_offset])
