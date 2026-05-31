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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    # Get the batch and head indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Create a grid of M and N dimensions
    m_block = tl.program_id(2)
    n_block = tl.program_id(3)
    
    # Compute offsets for Q
    q_offset = batch_idx * stride_qb + head_idx * stride_qh + m_block * stride_qm
    k_offset = batch_idx * stride_kb + head_idx * stride_kh + n_block * stride_km
    v_offset = batch_idx * stride_vb + head_idx * stride_vh + n_block * stride_vm
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k_block in range(0, (head_dim + BLOCK_K - 1) // BLOCK_K):
        # Load Q, K, V tiles
        q = tl.load(Q_ptr + q_offset + k_block * stride_qk, mask=(k_block * BLOCK_K + tl.arange(0, BLOCK_K)) < head_dim)
        k = tl.load(K_ptr + k_offset + k_block * stride_kk, mask=(k_block * BLOCK_K + tl.arange(0, BLOCK_K)) < head_dim)
        v = tl.load(V_ptr + v_offset + k_block * stride_vk, mask=(k_block * BLOCK_K + tl.arange(0, BLOCK_K)) < head_dim)
        
        # Compute attention score
        qk = tl.sum(q[:, None] * k[None, :], axis=2)
        
        # Apply scaling
        qk = qk / tl.sqrt(head_dim)
        
        # Apply causal masking if needed
        if IS_CAUSAL:
            mask = (m_block * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) >= (n_block * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
            qk = tl.where(mask, qk, float('-inf'))
        
        # Apply softmax
        qk = qk - tl.max(qk, axis=1, keepdims=True)
        exp_qk = tl.exp(qk)
        sum_exp_qk = tl.sum(exp_qk, axis=1, keepdims=True)
        probs = exp_qk / sum_exp_qk
        
        # Accumulate
        acc += probs @ v
    
    # Store output
    out_offset = batch_idx * stride_ob + head_idx * stride_oh + m_block * stride_om
    tl.store(Out_ptr + out_offset, acc, mask=(m_block * BLOCK_M + tl.arange(0, BLOCK_M)) < seq_len)

def triton_scaled_dot_product_attention(Q, K, V, is_causal=False):
    """
    Triton implementation of scaled dot product attention.
    """
    # Ensure inputs are contiguous and on GPU
    Q = Q.contiguous().to(torch.float32)
    K = K.contiguous().to(torch.float32)
    V = V.contiguous().to(torch.float32)
    
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Create output tensor
    Out = torch.empty_like(Q)
    
    # Define block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    # Grid configuration
    grid = (
        batch_size,
        num_heads,
        (seq_len + BLOCK_M - 1) // BLOCK_M,
        (seq_len + BLOCK_N - 1) // BLOCK_N
    )
    
    # Launch kernel
    scaled_dot_product_attention_kernel[grid](
        Q, K, V, Out,
        seq_len, head_dim,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        BLOCK_M, BLOCK_N, BLOCK_K,
        is_causal
    )
    
    return Out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Use our Triton kernel instead of PyTorch's implementation
        return triton_scaled_dot_product_attention(Q, K, V)