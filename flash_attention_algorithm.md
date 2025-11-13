# Flash Attention Algorithms

## Algorithm 1: Flash Attention
'''
Tối ưu việc tính attention cho sequences dài, giữ độ chính xác.
'''
**Input:** 
- Q, K, V: Separate tensors, dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic

**Initialize:**
- softmax_scale ← 1/√(headdim) if None, BLOCK_M ← 128, BLOCK_N ← 128
- num_groups ← nheads // nheads_kv (for MQA/GQA support)
- output ← tensor of zeros with same shape as Q

**for** batch_idx, head_idx **do**
    kv_head_idx ← head_idx // num_groups
    
    **for** i = 0 to seqlen by BLOCK_M **do**
        // Divide queries into small tiles and initialize accumulators
        q_tile ← extract query tile from position i to i+BLOCK_M
        initialize output accumulator (o_i), normalizer (l_i), and max tracker (m_i)
        
        **for** j = 0 to seqlen by BLOCK_N **do**
            // Divide keys and values into corresponding small tiles
            k_tile, v_tile ← extract key-value tiles from position j to j+BLOCK_N
            
            // Compute attention scores between query tile and key tile
            S_ij ← compute dot product (q_tile × k_tile transpose) and multiply by scale
            
            // Apply various masks to restrict attention
            apply causal mask (if needed), sliding window mask, and ALiBi bias
            
            // Update results using online softmax algorithm
            update accumulator with current tile, adjust normalizer and max tracker
        **end for**
        
        // Normalize and store final results for this query tile
        output[batch_idx, i:i+BLOCK_M, head_idx, :] ← normalize o_i by l_i
    **end for**
**end for**

**Return:** output

---

## Algorithm 2: Flash Attention with KV Cache
'''
Design cho text generation và inference trong transformer model, mà KV cache được sử dụng để tối ưu hóa việc tính toán attention
'''
**Input:**
- Q: Query tensor, K_cache, V_cache: Cached tensors
- K, V: New tensors (optional), rotary_cos, rotary_sin: RoPE (optional)
- cache_seqlens: Cache lengths, softmax_scale, causal, window_size, alibi_slopes

**Initialize:**
- softmax_scale ← 1/√(headdim) if None

**if** K, V provided **then**
    **if** RoPE provided **then** K, V ← apply_rotary_embedding(K, V, rotary_cos, rotary_sin)
    Update K_cache, V_cache with new K, V
**end if**

output ← flash_attn_func(Q, K_cache, V_cache, softmax_scale, causal, window_size, alibi_slopes)

**Return:** output

---

Ví Dụ Thực Tế:
Giả sử đang generate text "Hello world":

Step 1: Generate "Hello"

Q: embedding của "Hello"
K_cache, V_cache: trống
Output: hidden state cho "Hello"
Step 2: Generate "world"

Q: embedding của "world"
K_cache, V_cache: chứa K,V của "Hello"
K, V: K,V mới của "world"
Update cache và tính attention

## Giải thích các hàm tensor cơ bản:

### Tensor Creation Functions:
- `zeros(shape)`: Tạo tensor với các giá trị 0 với shape cho trước
- `tensor of zeros with same shape and dtype as X`: Tạo tensor 0 có cùng kích thước và kiểu dữ liệu với X
- `tensor filled with value`: Tạo tensor với tất cả phần tử bằng một giá trị cụ thể
- `ones(shape)`: Tạo tensor với các giá trị 1
- `sequence_from_a_to_b`: Tạo sequence từ a đến b (tương đương arange trong NumPy/PyTorch)

### Tensor Operations:
- `upper_triangular_matrix_of_ones`: Tạo ma trận tam giác trên với các giá trị 1
- `fill_masked_positions(tensor, mask, value)`: Điền giá trị vào các vị trí được mask
- `concatenate([tensor1, tensor2], dim)`: Nối các tensor theo chiều dim
- `chunk(tensor, chunks, dim)`: Chia tensor thành chunks phần theo chiều dim
- `maximum(a, b)`: Lấy giá trị lớn nhất element-wise giữa hai tensor
- `exp(x)`: Hàm exponential element-wise

### Implementation in PyTorch:
```python
# zeros_like(x) tương đương:
torch.zeros_like(x)
# hoặc:
torch.zeros(x.shape, dtype=x.dtype, device=x.device)

# full(value, shape) tương đương:
torch.full(shape, value)

# arange(start, end) tương đương:
torch.arange(start, end)
```

---

## Supporting Functions:

### apply_rotary_embedding(K, V, cos, sin, interleaved):
'''
Thêm thông tin vị trí (positional info) của token vào K, V thông qua phép quay(rotation)
Input: K, V, tensor cos, tensor sin, interleaved(True/False)
Output: K_rot, V_rot - tensor K, V đã quay
'''
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