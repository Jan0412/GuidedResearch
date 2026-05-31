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
    PRE_BLOCK: tl.constexpr, STORE_TRANS: tl.constexpr
):
    # Get the block ID for this program
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Initialize accumulator for attention scores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the blocks of K and V
    for k_block in range(0, (head_dim + BLOCK_K - 1) // BLOCK_K):
        # Load Q tile
        q_offset = batch_id * stride_qb + head_id * stride_qh + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, BLOCK_K)[None, :] * stride_qk
        q = tl.load(Q_ptr + q_offset, mask=(tl.arange(0, BLOCK_M)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < head_dim), other=0.0)
        
        # Load K tile
        k_offset = batch_id * stride_kb + head_id * stride_kh + tl.arange(0, BLOCK_K)[:, None] * stride_km + tl.arange(0, BLOCK_N)[None, :] * stride_kk
        k = tl.load(K_ptr + k_offset, mask=(tl.arange(0, BLOCK_K)[:, None] < head_dim) & (tl.arange(0, BLOCK_N)[None, :] < seq_len), other=0.0)
        
        # Compute attention scores
        qk = tl.dot(q, k, allow_tf32=False)
        qk = qk / tl.sqrt(head_dim)
        
        # Apply causal mask if needed (for self-attention)
        # In this simplified version, we assume no causal masking
        
        # Apply softmax
        qk_max = tl.max(qk, axis=1, keepdims=True)
        qk = qk - qk_max
        qk = tl.exp(qk)
        qk_sum = tl.sum(qk, axis=1, keepdims=True)
        qk = qk / qk_sum
        
        # Load V tile
        v_offset = batch_id * stride_vb + head_id * stride_vh + tl.arange(0, BLOCK_N)[:, None] * stride_vm + tl.arange(0, BLOCK_K)[None, :] * stride_vk
        v = tl.load(V_ptr + v_offset, mask=(tl.arange(0, BLOCK_N)[:, None] < seq_len) & (tl.arange(0, BLOCK_K)[None, :] < head_dim), other=0.0)
        
        # Accumulate results
        acc += tl.dot(qk, v, allow_tf32=False)
    
    # Write output
    out_offset = batch_id * stride_ob + head_id * stride_oh + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, BLOCK_N)[None, :] * stride_ok
    tl.store(Out_ptr + out_offset, acc, mask=(tl.arange(0, BLOCK_M)[:, None] < seq_len) & (tl.arange(0, BLOCK_N)[None, :] < seq_len))

def triton_scaled_dot_product_attention(Q, K, V):
    """
    Custom Triton implementation of scaled dot product attention.
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "All tensors must be on CUDA."
    assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16, "All tensors must be float16."
    
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Prepare output tensor
    Out = torch.empty_like(Q)
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    # Grid dimensions
    grid = (
        batch_size,
        num_heads
    )
    
    # Launch kernel
    scaled_dot_product_attention_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_M, BLOCK_N, BLOCK_K,
        PRE_BLOCK=1024,
        STORE_TRANS=False
    )
    
    return Out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Use custom Triton kernel instead of torch.nn.functional.scaled_dot_product_attention
        return triton_scaled_dot_product_attention(Q, K, V)