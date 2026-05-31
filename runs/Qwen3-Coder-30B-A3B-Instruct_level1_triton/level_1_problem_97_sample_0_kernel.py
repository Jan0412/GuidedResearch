import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def scaled_dot_product_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SCALE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    m_id = tl.program_id(2)
    
    # Initialize accumulator for attention scores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k_id in range(0, (head_dim + BLOCK_K - 1) // BLOCK_K):
        # Load Q, K, V tiles
        offs_m = m_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = k_id * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Load Q tile (BLOCK_M x BLOCK_K)
        q_ptrs = Q_ptr + batch_id * stride_qb + head_id * stride_qh + \
                 offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim), other=0.0)
        
        # Load K tile (BLOCK_K x BLOCK_N)
        k_ptrs = K_ptr + batch_id * stride_kb + head_id * stride_kh + \
                 offs_k[:, None] * stride_km + offs_n[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(offs_k[:, None] < head_dim) & (offs_n[None, :] < seq_len), other=0.0)
        
        # Compute Q @ K^T
        qk = tl.dot(q, k, allow_tf32=False)
        qk *= SCALE
        
        # Apply causal mask if needed (can be extended for causal masking)
        # For this implementation, we assume no causal masking
        
        # Compute attention weights
        # Apply softmax (we'll do it in chunks to avoid overflow)
        max_val = tl.max(qk, axis=1, keepdims=True)
        qk = qk - max_val
        exp_qk = tl.exp(qk)
        sum_exp = tl.sum(exp_qk, axis=1, keepdims=True)
        attn_weights = exp_qk / sum_exp
        
        # Load V tile (BLOCK_N x BLOCK_K)
        v_ptrs = V_ptr + batch_id * stride_vb + head_id * stride_vh + \
                 offs_n[:, None] * stride_vm + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len) & (offs_k[None, :] < head_dim), other=0.0)
        
        # Compute attention output
        out = tl.dot(attn_weights, v, allow_tf32=False)
        
        # Accumulate
        acc += out
    
    # Write output
    offs_m = m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    out_ptrs = Out_ptr + batch_id * stride_ob + head_id * stride_oh + \
               offs_m[:, None] * stride_om + offs_n[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len))

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        
        # Get dimensions
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        # Scale factor for scaled dot product attention
        scale = 1.0 / math.sqrt(head_dim)
        
        # Allocate output tensor
        out = torch.empty_like(Q)
        
        # Define block sizes
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 64
        
        # Grid dimensions
        grid = (
            batch_size,
            num_heads,
            (seq_len + BLOCK_M - 1) // BLOCK_M
        )
        
        # Launch kernel
        scaled_dot_product_attention_kernel[grid](
            Q, K, V, out,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch_size, num_heads, seq_len, head_dim,
            BLOCK_M, BLOCK_N, BLOCK_K,
            scale,
            num_warps=4
        )
        
        return out