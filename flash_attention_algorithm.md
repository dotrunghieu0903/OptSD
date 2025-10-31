# Flash Attention Algorithms

## Algorithm 1: Flash Attention QKV Packed

**Input:** 
- QKV tensor (batch_size, seqlen, 3, nheads, headdim)
- dropout_p: Dropout probability
- causal: Boolean for causal masking
- window_size: Tuple (left, right) for sliding window
- alibi_slopes: ALiBi bias slopes
- deterministic: Boolean for deterministic backward pass

**Initialize:** 
- Extract Q, K, V from packed tensor: Q, K, V ← QKV[:,:,0,:,:], QKV[:,:,1,:,:], QKV[:,:,2,:,:]
- softmax_scale ← 1/√(headdim)
- BLOCK_M ← 128, BLOCK_N ← 128
- output ← zeros_like(Q)

**for** i = 0, BLOCK_M, 2×BLOCK_M, ... , seqlen **do**
    q_tile ← Q[:,i:i+BLOCK_M,:,:]
    o_i ← zeros_like(q_tile)
    l_i ← zeros(batch_size, nheads, BLOCK_M)
    m_i ← full(-∞, batch_size, nheads, BLOCK_M)
    
    **for** j = 0, BLOCK_N, 2×BLOCK_N, ... , seqlen **do**
        k_tile ← K[:,j:j+BLOCK_N,:,:]
        v_tile ← V[:,j:j+BLOCK_N,:,:]
        
        S_ij ← (q_tile @ k_tile^T) × softmax_scale
        
        **if** causal **then**
            Apply causal mask to S_ij
        **end if**
        
        **if** window_size ≠ (-1,-1) **then**
            Apply sliding window mask to S_ij
        **end if**
        
        **if** alibi_slopes is not None **then**
            Apply ALiBi bias to S_ij
        **end if**
        
        m_ij ← max(S_ij, dim=-1)
        m_new ← maximum(m_i, m_ij)
        
        o_i ← o_i × exp(m_i - m_new)
        l_i ← l_i × exp(m_i - m_new)
        
        P_ij ← exp(S_ij - m_new)
        o_i ← o_i + P_ij @ v_tile
        l_i ← l_i + sum(P_ij, dim=-1)
        m_i ← m_new
    **end for**
    
    output[:,i:i+BLOCK_M,:,:] ← o_i / l_i
**end for**

**if** dropout_p > 0 and training **then**
    output ← dropout(output, p=dropout_p)
**end if**

**Return:** output

---

## Algorithm 2: Flash Attention Separate QKV

**Input:** 
- Q: Query tensor (batch_size, seqlen, nheads, headdim)
- K: Key tensor (batch_size, seqlen, nheads_kv, headdim)  
- V: Value tensor (batch_size, seqlen, nheads_kv, headdim)
- dropout_p: Dropout probability
- softmax_scale: Scaling factor for attention scores
- causal: Boolean for causal masking
- window_size: Tuple (left, right) for sliding window
- alibi_slopes: ALiBi bias slopes
- deterministic: Boolean for deterministic implementation

**Initialize:**
- **if** softmax_scale is None **then** softmax_scale ← 1/√(headdim)
- BLOCK_M ← 128, BLOCK_N ← 128
- batch_size, seqlen, nheads, headdim ← Q.shape
- nheads_kv ← K.shape[2]
- num_groups ← nheads // nheads_kv
- output ← zeros_like(Q)

**for** batch_idx = 0, 1, ..., batch_size-1 **do**
    **for** head_idx = 0, 1, ..., nheads-1 **do**
        kv_head_idx ← head_idx // num_groups
        
        **for** i = 0, BLOCK_M, 2×BLOCK_M, ..., seqlen **do**
            q_tile ← Q[batch_idx, i:i+BLOCK_M, head_idx, :]
            o_i ← zeros_like(q_tile)
            l_i ← zeros(BLOCK_M)
            m_i ← full(-∞, BLOCK_M)
            
            **for** j = 0, BLOCK_N, 2×BLOCK_N, ..., seqlen **do**
                k_tile ← K[batch_idx, j:j+BLOCK_N, kv_head_idx, :]
                v_tile ← V[batch_idx, j:j+BLOCK_N, kv_head_idx, :]
                
                S_ij ← (q_tile @ k_tile^T) × softmax_scale
                
                **if** causal **then**
                    mask ← triu(ones(BLOCK_M, BLOCK_N), diagonal=j-i+1)
                    S_ij ← masked_fill(S_ij, mask, -∞)
                **end if**
                
                **if** window_size ≠ (-1,-1) **then**
                    left_window, right_window ← window_size
                    row_pos ← arange(i, i+BLOCK_M)
                    col_pos ← arange(j, j+BLOCK_N)
                    window_mask ← (col_pos < row_pos - left_window) | (col_pos > row_pos + right_window)
                    S_ij ← masked_fill(S_ij, window_mask, -∞)
                **end if**
                
                **if** alibi_slopes is not None **then**
                    distance ← |row_pos - col_pos|
                    alibi_bias ← -alibi_slopes[head_idx] × distance
                    S_ij ← S_ij + alibi_bias
                **end if**
                
                m_ij ← max(S_ij, dim=-1)
                m_new ← maximum(m_i, m_ij)
                
                exp_diff_old ← exp(m_i - m_new)
                o_i ← o_i × exp_diff_old
                l_i ← l_i × exp_diff_old
                
                P_ij ← exp(S_ij - m_new)
                o_i ← o_i + P_ij @ v_tile
                l_i ← l_i + sum(P_ij, dim=-1)
                m_i ← m_new
            **end for**
            
            output[batch_idx, i:i+BLOCK_M, head_idx, :] ← o_i / l_i
        **end for**
    **end for**
**end for**

**if** dropout_p > 0 and training **then**
    output ← dropout(output, p=dropout_p)
**end if**

**Return:** output

---

## Algorithm 3: Flash Attention with KV Cache

**Input:**
- Q: Query tensor (batch_size, seqlen_q, nheads, headdim)
- K_cache: Cached key tensor (batch_size, seqlen_cache, nheads_kv, headdim)
- V_cache: Cached value tensor (batch_size, seqlen_cache, nheads_kv, headdim)
- K: New key tensor (optional)
- V: New value tensor (optional)
- rotary_cos, rotary_sin: RoPE embeddings (optional)
- cache_seqlens: Cache sequence lengths
- cache_batch_idx: Cache batch indices
- block_table: Block table for paged attention
- softmax_scale: Scaling factor
- causal: Boolean for causal masking
- window_size: Sliding window size
- rotary_interleaved: Boolean for RoPE interleaving
- alibi_slopes: ALiBi bias slopes

**Initialize:**
- **if** softmax_scale is None **then** softmax_scale ← 1/√(headdim)
- batch_size, seqlen_q, nheads, headdim ← Q.shape

**if** K is not None and V is not None **then**
    **if** rotary_cos is not None and rotary_sin is not None **then**
        K, V ← apply_rotary_embedding(K, V, rotary_cos, rotary_sin, rotary_interleaved)
    **end if**
    
    **if** cache_seqlens is not None **then**
        **for** i = 0, 1, ..., batch_size-1 **do**
            seq_len ← cache_seqlens[i]
            K_cache[i, seq_len:seq_len+seqlen_new] ← K[i]
            V_cache[i, seq_len:seq_len+seqlen_new] ← V[i]
        **end for**
    **else**
        K_cache ← concatenate([K_cache, K], dim=1)
        V_cache ← concatenate([V_cache, V], dim=1)
    **end if**
**end if**

**if** block_table is not None **then**
    Apply paged attention with block_table
**end if**

output ← flash_attn_func(Q, K_cache, V_cache, 0.0, softmax_scale, causal, window_size, alibi_slopes, False)

**Return:** output

---

## Supporting Functions:

### apply_rotary_embedding(K, V, cos, sin, interleaved):
**if** interleaved **then**
    K_rot ← K × cos + rotate_half(K) × sin
    V_rot ← V × cos + rotate_half(V) × sin
**else**
    d ← headdim // 2
    K_rot ← concatenate([K[..., :d] × cos - K[..., d:] × sin, 
                        K[..., :d] × sin + K[..., d:] × cos], dim=-1)
    V_rot ← concatenate([V[..., :d] × cos - V[..., d:] × sin,
                        V[..., :d] × sin + V[..., d:] × cos], dim=-1)
**end if**
**Return:** K_rot, V_rot

### rotate_half(x):
x1, x2 ← chunk(x, 2, dim=-1)
**Return:** concatenate([-x2, x1], dim=-1)