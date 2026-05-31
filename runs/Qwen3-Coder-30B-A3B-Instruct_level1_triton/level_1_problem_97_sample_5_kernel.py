import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def scaled_dot_product_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seq_len, head_dim,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    SCALE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Compute the starting positions for this program
    m_start = tl.program_id(2) * BLOCK_M
    
    # Loop over the sequence dimension in chunks
    for m_off in range(m_start, seq_len, BLOCK_M):
        # Initialize accumulator for attention scores
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Loop over the key/value dimension in chunks
        for k_off in range(0, head_dim, BLOCK_K):
            # Load Q fragment
            q = tl.load(Q_ptr + 
                       batch_idx * stride_qb + 
                       head_idx * stride_qh + 
                       m_off * stride_qm + 
                       k_off * stride_qk,
                       mask=(m_off + tl.arange(0, BLOCK_M)[:, None] < seq_len) &
                            (k_off + tl.arange(0, BLOCK_K)[None, :] < head_dim),
                       other=0.0)
            
            # Load K fragment
            k = tl.load(K_ptr + 
                       batch_idx * stride_kb + 
                       head_idx * stride_kh + 
                       (k_off // BLOCK_K) * BLOCK_K * stride_km + 
                       tl.arange(0, BLOCK_K)[:, None] * stride_kk,
                       mask=(k_off + tl.arange(0, BLOCK_K)[:, None] < head_dim) &
                            (tl.arange(0, BLOCK_N)[None, :] < seq_len),
                       other=0.0)
            
            # Compute attention score
            attn_scores = tl.dot(q, k, trans_b=True) * SCALE
            
            # Apply causal mask if needed
            # For simplicity, we assume no causal masking in this implementation
            # In practice, you would add a mask here
            
            # Apply softmax (online softmax for numerical stability)
            max_val = tl.max(attn_scores, axis=1, keep_dims=True)
            exp_scores = tl.exp(attn_scores - max_val)
            sum_exp = tl.sum(exp_scores, axis=1, keep_dims=True)
            probs = exp_scores / sum_exp
            
            # Load V fragment
            v = tl.load(V_ptr + 
                       batch_idx * stride_vb + 
                       head_idx * stride_vh + 
                       (k_off // BLOCK_K) * BLOCK_K * stride_vm + 
                       tl.arange(0, BLOCK_K)[:, None] * stride_vk,
                       mask=(k_off + tl.arange(0, BLOCK_K)[:, None] < head_dim) &
                            (tl.arange(0, BLOCK_N)[None, :] < seq_len),
                       other=0.0)
            
            # Accumulate attention result
            acc += tl.dot(probs, v)
        
        # Store output
        out_ptrs = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh + m_off * stride_om
        tl.store(out_ptrs,
                acc,
                mask=m_off + tl.arange(0, BLOCK_M)[:, None] < seq_len)

# Optimized scaled dot product attention using Triton
def triton_scaled_dot_product_attention(Q, K, V):
    # Ensure tensors are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Calculate scaling factor
    scale = 1.0 / math.sqrt(head_dim)
    
    # Create output tensor
    Out = torch.empty_like(Q)
    
    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    # Grid dimensions
    grid = (
        batch_size,
        num_heads,
        (seq_len + BLOCK_M - 1) // BLOCK_M
    )
    
    # Launch kernel
    scaled_dot_product_attention_kernel[grid](
        Q, K, V, Out,
        seq_len, head_dim,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SCALE=scale
    )
    
    return Out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = triton_scaled_dot_product_attention(Q, K, V)
        return out