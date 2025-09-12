# Problem 1: FlashAttention-2 Tiled Forward Pass Implementation in PyTorch
# Implement the forward pass of FlashAttention-2 in PyTorch.
# This is a simplified version for educational purposes.
# The implementation should follow the algorithm described in the FlashAttention-2 paper.
# Reference: https://arxiv.org/abs/2307.08691
import torch
import torch.nn as nn
import math

import triton
import triton.language as tl
from typing import Optional

class FlashAttention2Function(torch.autograd.Function):
    """
    A pure PyTorch implementation of the FlashAttention-2 forward pass.
    This version is a template for student implementation.
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
                        
                        # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
                        # 1. Apply causal masking if is_causal is True
                        if is_causal:
                            row_ids = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                            col_ids = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                            causal_mask = row_ids < col_ids
                            S_ij = torch.where(causal_mask, float('-inf'), S_ij)
                        
                        # 2. Compute the new running maximum
                        m_new = torch.maximum(m_i.unsqueeze(1), S_ij.max(dim=1)[0])
                        
                        # 3. Rescale the previous accumulators
                        exp_scale = torch.exp(m_i - m_new)
                        o_i = o_i * exp_scale.unsqueeze(1)
                        l_i = l_i * exp_scale
                        
                        # 4. Compute the probabilities for the current tile
                        P_tilde_ij = torch.exp(S_ij - m_new.unsqueeze(1))
                        
                        # 5. Accumulate the current tile's contribution
                        l_i = l_i + P_tilde_ij.sum(dim=1)
                        o_i = o_i + P_tilde_ij @ V_tile
                        
                        # 6. Update the running max for the next iteration
                        m_i = m_new
                        
                        # --- END OF STUDENT IMPLEMENTATION ---

                    # After iterating through all key tiles, normalize the output
                    # This part is provided for you. It handles the final division safely.
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
        raise NotImplementedError("Backward pass not yet implemented for FlashAttention2Function")
    
# Problem 2: Triton Basics
# Triton Kernel for Weighted Row-Sum
# Implement a Triton kernel to compute the weighted sum of each row in a matrix.
# The operation is defined as:
# Y[i] = sum_{j=0}^{N_COLS-1} X[i, j] * W[j]
# where X is a 2D tensor of shape (N_ROWS, N_COLS) and W is a 1D tensor of shape (N_COLS).
# The output Y is a 1D tensor of shape (N_ROWS).
# Reference: https://triton-lang.org/


@triton.jit
def weighted_row_sum_kernel(
    X_ptr,        # Pointer to the input tensor
    W_ptr,        # Pointer to the weight vector
    Y_ptr,        # Pointer to the output vector
    N_COLS,       # Number of columns in the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size for the kernel
):
    """
    Triton kernel to compute the weighted sum of each row in a matrix.
    Y[i] = sum_{j=0}^{N_COLS-1} X[i, j] * W[j]
    """
    # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
    # 1. Get the row index for the current program instance
    row_idx = tl.program_id(axis=0)

    # 2. Create a pointer to the start of the current row in the input tensor X
    row_start_ptr = X_ptr + row_idx * N_COLS
    
    # 3. Create a pointer for the output vector Y
    output_ptr = Y_ptr + row_idx
    
    # 4. Initialize an accumulator for the sum of the products
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # 5. Iterate over the columns in blocks
    num_blocks = tl.cdiv(N_COLS, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        # Calculate column offsets for the current block
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for valid columns
        mask = col_offsets < N_COLS
        
        # Load data chunks safely using the mask
        x_chunk = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        w_chunk = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        
        # Compute element-wise product and accumulate
        accumulator += x_chunk * w_chunk
        
    # 6. Reduce the accumulator to a single value
    final_sum = tl.sum(accumulator)
    
    # 7. Store the final sum to the output
    tl.store(output_ptr, final_sum)
    
    # --- END OF STUDENT IMPLEMENTATION ---
def weighted_row_sum_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for the weighted row-sum operation using the Triton kernel.
    """
    assert x.dim() == 2, "Input tensor must be a 2D matrix"
    assert w.dim() == 1, "Weight tensor must be a 1D vector"
    assert x.shape[1] == w.shape[0], "Inner dimensions must match"
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA"
    
    N_ROWS, N_COLS = x.shape
    
    # The output is a 1D tensor with length equal to the number of rows.
    y = torch.empty(N_ROWS, device=x.device, dtype=torch.float32)
    
    # The grid is 1D, with one program instance per row.
    grid = (N_ROWS,)
    
    # Block size is a power of 2. 1024 is a good default.
    BLOCK_SIZE = 1024
    
    # Launch the kernel
    weighted_row_sum_kernel[grid](
        x, w, y,
        N_COLS=N_COLS,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y.to(x.dtype) # Cast back to original dtype

def torch_weighted_row_sum(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using pure PyTorch.
    """
    return (x * w).sum(dim=1)

# Problem 3: Triton Kernel for FlashAttention
# Implement the forward pass of FlashAttention-2 using Triton.
# Reference: https://triton-lang.org/

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
    N_HEADS,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of FlashAttention-2 (non-causal).
    This is a template for student implementation.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_HEADS
    head_idx = batch_head_idx % N_HEADS

    # 2. Initialize pointers and accumulators for the online softmax.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    # PyTorch softmax is exp(x), Triton is exp2(x * log2(e)), log2(e) is approx 1.44269504
    qk_scale = softmax_scale * 1.44269504

    # 4. Main loop: Iterate over blocks of keys (K_j) and values (V_j).
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        # - Load K_j
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
        # 1. Find the new running maximum for numerical stability
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        
        # 2. Rescale the existing accumulator and denominator
        exp_scale = tl.exp2((m_i - m_new) * qk_scale)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        # 3. Compute attention probabilities for current tile
        p_ij = tl.exp2(s_ij * qk_scale - m_new[:, None])
        
        # 4. Update accumulator using p_ij and v_block
        acc += tl.dot(p_ij, v_block)
        
        # 5. Update denominator (row sum of probabilities)
        l_i += tl.sum(p_ij, axis=1)
        
        # 6. Update running maximum for next iteration
        m_i = m_new
        # --- END OF STUDENT IMPLEMENTATION ---


    # 5. Normalize the accumulator and write the output block.
    # This part is provided. It handles the final normalization and write-back.
    l_i_safe = l_i[:, None] + 1e-6
    acc = acc / l_i_safe
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)

def flash_attention_forward(q, k, v, is_causal=False):
    """
    Minimal Python wrapper for the FlashAttention-2 forward pass.
    """
    batch, n_heads, seq_len, head_dim = q.shape
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

    _flash_attention_forward_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o

# Problem 4: Adding Causal Masking to the Triton Kernel
# Extend the Triton kernel from Problem 3 to support causal masking.
# When `is_causal` is True, ensure that each query position can only attend to
# key positions that are less than or equal to its own position.
# Reference: https://triton-lang.org/
@triton.jit
def _flash_attention_forward_causal_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_HEADS,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention.
    This is a template for student implementation.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_HEADS
    head_idx = batch_head_idx % N_HEADS

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    # PyTorch softmax is exp(x), Triton is exp2(x * log2(e)), log2(e) is approx 1.44269504
    qk_scale = softmax_scale * 1.44269504

    # --- Phase 1: Accumulate in Off-Diagonal Blocks (No Masking) ---
    # Process key/value blocks that are strictly in the past (q_idx > k_idx).
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
        # Load K_j block - these are blocks that are strictly before the query blocks,
        # so we don't need causal masking as all keys are already in the past
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        # Compute attention scores
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        
        # Load V_j block
        v_ptrs = V_ptr + batch_idx * v_stride_b + head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Update running maximum and rescale accumulators
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        exp_scale = tl.exp2((m_i - m_new) * 1.0)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        # Compute attention probabilities and update accumulators
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
        m_i = m_new
        # --- END OF STUDENT IMPLEMENTATION ---


    # --- Phase 2: Run on the Diagonal Blocks (With Masking) ---
    # Process the blocks where query and key indices can overlap.
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
        # Load K_j block for diagonal block (where causal masking is needed)
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        # Compute attention scores
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        
        # Apply causal mask: row_idx >= col_idx
        # Get query and key indices
        q_pos = q_offsets[:, None]
        k_pos = k_offsets[None, :]
        
        # Create mask where each query can only attend to keys at positions <= query position
        causal_mask = q_pos >= k_pos
        
        # Apply mask by replacing masked positions with -inf
        s_ij = tl.where(causal_mask, s_ij, float('-inf'))
        
        # Load V_j block
        v_ptrs = V_ptr + batch_idx * v_stride_b + head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Update running maximum and rescale accumulators
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        exp_scale = tl.exp2((m_i - m_new) * 1.0)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        # Compute attention probabilities and update accumulators
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
        m_i = m_new
        # --- END OF STUDENT IMPLEMENTATION ---


    # 4. Normalize and write the final output block.
    l_i_safe = l_i[:, None] + 1e-6
    acc = acc / l_i_safe
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)

def flash_attention_forward(q, k, v, is_causal=True):
    """
    Python wrapper for the single-kernel, two-phase causal FlashAttention.
    """
    if not is_causal:
        raise NotImplementedError("This implementation is for causal attention. Use solution_3 for non-causal.")

    batch, n_heads, seq_len, head_dim = q.shape
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

    _flash_attention_forward_causal_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o

# Problem 5: Extending for Grouped-Query Attention (GQA)
# Extend the Triton kernel from Problem 4 to support Grouped-Query Attention (GQA).
# In GQA, the number of query heads (H_q) can be different from the number of key/value heads (H_kv).
# Each query head attends to a specific group of key/value heads.
# Reference: https://triton-lang.org/

import torch
import triton
import triton.language as tl
import math

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
        exp_scale = tl.exp2((m_i - m_new) * 1.0)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        # Compute attention probabilities and update accumulators
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
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
        causal_mask = q_pos >= k_pos
        s_ij = tl.where(causal_mask, s_ij, float('-inf'))
        
        # Load V_j block using kv_head_idx
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Online softmax update with causal masking (reused from Problem 4)
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        exp_scale = tl.exp2((m_i - m_new) * 1.0)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        # Compute attention probabilities and update accumulators
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
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

# Problem 6+7: Extending for Sliding Window Attention(SWA) and Sink Attention(StreamingLLM)
# Extend the Triton kernel from Problem 5 to support Sliding Window Attention and Sink Attention.
# In Sliding Window Attention, each query position attends to a fixed-size window of key positions around it.
# In Sink Attention, certain "sink" positions can attend to all previous key positions.
# Reference: https://triton-lang.org/
# Problem 6
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
    window_start = tl.maximum(0, q_block_idx * BLOCK_M - WINDOW_SIZE + BLOCK_N)

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
        window_mask = (q_pos - k_pos) < WINDOW_SIZE
        s_ij = tl.where(window_mask, s_ij, float('-inf'))
        
        # Load V_j block
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        exp_scale = tl.exp2((m_i - m_new) * qk_scale)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        p_ij = tl.exp2(s_ij * qk_scale - m_new[:, None])
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
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
        window_mask = (q_pos - k_pos) < WINDOW_SIZE  # Sliding window masking
        mask = causal_mask & window_mask
        s_ij = tl.where(mask, s_ij, float('-inf'))
        
        # Load V_j block
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Online softmax update with masked attention scores
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        exp_scale = tl.exp2((m_i - m_new) * qk_scale)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        p_ij = tl.exp2(s_ij * qk_scale - m_new[:, None])
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
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

# Problem 7: Sink Attention
# Extend the Triton kernel from Problem 6 to support Sink Attention.
# In Sink Attention, certain "sink" positions can attend to all previous key positions.
# Reference: https://triton-lang.org/

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
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

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

    # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
    # Phase 0: Process Sink Tokens (always attend to these regardless of window)
    if SINK_SIZE > 0:
        for start_n in range(0, SINK_SIZE, BLOCK_N):
            # Load K_j block for sink tokens
            k_offsets = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                     (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
            k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SINK_SIZE, other=0.0)
            
            # Compute attention scores for sink tokens
            s_ij = tl.dot(q_block, k_block)
            s_ij *= qk_scale
            
            # Load V_j block
            v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                     (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
            v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SINK_SIZE, other=0.0)
            
            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
            exp_scale = tl.exp2((m_i - m_new) * 1.0)
            acc = acc * exp_scale[:, None]
            l_i = l_i * exp_scale
            
            p_ij = tl.exp2(s_ij - m_new[:, None])
            acc += tl.dot(p_ij, v_block)
            l_i += tl.sum(p_ij, axis=1)
            m_i = m_new

    # Phase 1: Off-Diagonal Blocks (within sliding window, after sink tokens)
    window_start = max(SINK_SIZE, q_block_idx * BLOCK_M - WINDOW_SIZE + BLOCK_N)
    for start_n in range(SINK_SIZE, window_start, BLOCK_N):
        # Load K_j block
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        # Compute attention scores
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        
        # Load V_j block
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        exp_scale = tl.exp2((m_i - m_new) * 1.0)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
        m_i = m_new

    # Phase 2: Diagonal Blocks (with causal masking)
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, min((q_block_idx + 1) * BLOCK_M, SEQ_LEN), BLOCK_N):
        # Load K_j block
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        # Compute attention scores
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        
        # Apply causal masking
        q_pos = q_offsets[:, None]
        k_pos = k_offsets[None, :]
        causal_mask = q_pos >= k_pos
        s_ij = tl.where(causal_mask, s_ij, float('-inf'))
        
        # Load V_j block
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        
        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        exp_scale = tl.exp2((m_i - m_new) * 1.0)
        acc = acc * exp_scale[:, None]
        l_i = l_i * exp_scale
        
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
        m_i = m_new
    # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
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
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

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
        SINK_SIZE=sink_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o

# Problem 8: Backward Pass - with Recomputation
# Implement the backward pass for the FlashAttention-2 forward pass with causal masking.
# Use recomputation to save memory during the backward pass.
# Reference: https://triton-lang.org/

class FlashAttention2Function(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2, supports causal attention and GQA.
    """
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, softmax_scale: Optional[float] = None):
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        assert is_causal, "This kernel only supports causal attention"
        assert n_heads % n_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        o = torch.empty_like(q)
        M = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        
        _flash_attention_forward_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            seq_len,
            n_heads,
            n_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = n_heads
        ctx.num_kv_heads = n_kv_heads
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = ctx.num_kv_heads

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # Step 1: Compute delta = sum(dO * O)
        delta = torch.sum(do * o, dim=-1, keepdim=True)  # [batch, heads, seq_len, 1]
        
        # Initialize gradient buffers
        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        
        # Launch backward kernel
        _flash_attention_backward_kernel[grid](
            q, k, v, o, M, delta, do,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            ctx.softmax_scale,
            batch,
            n_heads,
            n_kv_heads,
            seq_len,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        # Return gradients and None for non-tensor inputs
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None

def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)

# Problem 9: Backward Pass - with Stored Softmax Probabilities
# Implement the backward pass for the FlashAttention-2 forward pass with causal masking.
# This time, store the softmax probabilities during the forward pass to simplify the backward computation.
# Reference: https://triton-lang.org/

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

        _flash_attention_forward_swa_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            seq_len,
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

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

        batch, n_q_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        # Define block sizes and compute grid
        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)
        
        # Compute delta = dO * O for each query position
        delta = torch.sum(do * o, dim=-1, keepdim=True)  # [batch, heads, seq_len, 1]
        
        # Launch the backward kernel
        _flash_attention_backward_stored_probs_kernel[grid](
            q, k, v, do, M, delta,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            batch,
            n_q_heads,
            n_kv_heads,
            seq_len,
            window_size or seq_len,
            sink_size or 0,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None
    
def flash_swda_with_sink(q, k, v, window_size: int, sink_size: int = 0, is_causal: bool = True, scale: Optional[float] = None):
    return FlashSWDAWithSink.apply(q, k, v, window_size, sink_size, is_causal, scale)

@triton.jit
def _flash_attention_forward_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
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
    Forward pass of Flash Attention.
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

    # Loop over key blocks
    for k_block_idx in range(triton.cdiv(SEQ_LEN, BLOCK_N)):
        k_offsets = k_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load K_j block
        k_ptrs = K_ptr + (
            batch_idx * k_stride_b +
            kv_head_idx * k_stride_h +
            k_offsets[None, :] * k_stride_s +
            tl.arange(0, HEAD_DIM)[:, None]
        )
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        # Compute attention scores
        s_ij = tl.dot(q_block, k_block) * softmax_scale

        # Apply causal masking
        causal_mask = q_offsets[:, None] >= k_offsets[None, :]
        s_ij = tl.where(causal_mask, s_ij, float('-inf'))

        # Load V_j block
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
        
        # Update with current block
        p_ij = tl.exp(s_ij - m_new[:, None])
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
        m_i = m_new

    # Store the running max for backward pass
    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    tl.store(m_ptrs, m_i, mask=q_offsets < SEQ_LEN)
    
    # Normalize and store the final output
    acc = acc / l_i[:, None]
    o_ptrs = O_ptr + (
        batch_idx * o_stride_b +
        q_head_idx * o_stride_h +
        q_offsets[:, None] * o_stride_s +
        tl.arange(0, HEAD_DIM)[None, :]
    )
    tl.store(o_ptrs, acc, mask=q_offsets[:, None] < SEQ_LEN)

@triton.jit
def _flash_attention_backward_kernel(
    # In/Out Pointers
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr, Delta_ptr, dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    # Strides
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
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
    Backward pass of Flash Attention with recomputation.
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
        batch_idx * o_stride_b +
        q_head_idx * o_stride_h +
        q_offsets[:, None] * o_stride_s +
        tl.arange(0, HEAD_DIM)[None, :]
    )
    do_block = tl.load(do_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

    # Load delta for this query block
    delta_ptrs = Delta_ptr + (
        batch_idx * o_stride_b +
        q_head_idx * o_stride_h +
        q_offsets * o_stride_s
    )
    delta_block = tl.load(delta_ptrs, mask=q_offsets < SEQ_LEN, other=0.0)

    # Initialize gradient accumulators
    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Load the stored maximum values
    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    m_i = tl.load(m_ptrs, mask=q_offsets < SEQ_LEN, other=0.0)

    # Loop over key/value blocks
    for k_block_idx in range(triton.cdiv(SEQ_LEN, BLOCK_N)):
        k_offsets = k_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load K block
        k_ptrs = K_ptr + (
            batch_idx * k_stride_b +
            kv_head_idx * k_stride_h +
            k_offsets[None, :] * k_stride_s +
            tl.arange(0, HEAD_DIM)[:, None]
        )
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        # Load V block
        v_ptrs = V_ptr + (
            batch_idx * v_stride_b +
            kv_head_idx * v_stride_h +
            k_offsets[:, None] * v_stride_s +
            tl.arange(0, HEAD_DIM)[None, :]
        )
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # Recompute attention scores and probabilities
        s_ij = tl.dot(q_block, k_block) * softmax_scale
        causal_mask = q_offsets[:, None] >= k_offsets[None, :]
        s_ij = tl.where(causal_mask, s_ij, float('-inf'))
        p_ij = tl.exp(s_ij - m_i[:, None])

        # Compute gradients for Q
        dp_ij = (do_block @ v_block.transpose(1, 0) - delta_block[:, None]) * p_ij
        dq_acc += dp_ij @ k_block.transpose(1, 0) * softmax_scale

        # Compute gradients for K and V (need to handle GQA)
        if N_Q_HEADS > N_KV_HEADS:
            group_size = N_Q_HEADS // N_KV_HEADS
            dp_ij_sum = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
            for g in range(group_size):
                head_offset = kv_head_idx * group_size + g
                do_g_ptrs = dO_ptr + (
                    batch_idx * o_stride_b +
                    head_offset * o_stride_h +
                    q_offsets[:, None] * o_stride_s +
                    tl.arange(0, HEAD_DIM)[None, :]
                )
                do_g = tl.load(do_g_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
                delta_g_ptrs = Delta_ptr + (
                    batch_idx * o_stride_b +
                    head_offset * o_stride_h +
                    q_offsets * o_stride_s
                )
                delta_g = tl.load(delta_g_ptrs, mask=q_offsets < SEQ_LEN, other=0.0)
                dp_ij_sum += ((do_g @ v_block.transpose(1, 0) - delta_g[:, None]) * p_ij).transpose(1, 0)
            
            dk_acc += dp_ij_sum @ q_block * softmax_scale
            dv_acc += dp_ij_sum.transpose(1, 0) @ do_block
        else:
            dk_acc += dp_ij.transpose(1, 0) @ q_block * softmax_scale
            dv_acc += p_ij.transpose(1, 0) @ do_block

    # Store Q gradients
    dq_ptrs = dQ_ptr + (
        batch_idx * q_stride_b +
        q_head_idx * q_stride_h +
        q_offsets[:, None] * q_stride_s +
        tl.arange(0, HEAD_DIM)[None, :]
    )
    tl.store(dq_ptrs, dq_acc, mask=q_offsets[:, None] < SEQ_LEN)

    # Store K and V gradients (with atomic adds for GQA)
    for k_offset in range(0, BLOCK_N):
        if k_offset >= SEQ_LEN:
            break
            
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