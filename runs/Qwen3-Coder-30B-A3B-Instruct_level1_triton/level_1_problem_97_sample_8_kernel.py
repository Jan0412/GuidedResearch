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
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    m_block = tl.program_id(2)
    
    # Initialize pointers for Q
    q_ptrs = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh + m_block * stride_qm
    
    # Initialize pointers for K and V
    k_ptrs = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_ptrs = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    
    # Initialize accumulator for attention scores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the sequence dimension
    for k_block in range(0, (seq_len + BLOCK_K - 1) // BLOCK_K):
        # Load Q, K, V tiles
        q = tl.load(q_ptrs + k_block * BLOCK_K * stride_qk, mask=(k_block * BLOCK_K + tl.arange(0, BLOCK_K) < seq_len)[None, :], other=0.0)
        
        k = tl.load(k_ptrs + k_block * BLOCK_K * stride_kk, mask=(k_block * BLOCK_K + tl.arange(0, BLOCK_K) < seq_len)[:, None], other=0.0)
        
        # Compute attention scores
        attn_scores = tl.dot(q, k.T, allow_tf32=False)
        attn_scores *= SCALE
        
        # Apply causal mask if needed (can be extended)
        # For now assuming no causal masking
        
        # Apply softmax
        max_val = tl.max(attn_scores, axis=1, keepdims=True)
        exp_scores = tl.exp(attn_scores - max_val)
        sum_exp = tl.sum(exp_scores, axis=1, keepdims=True)
        probs = exp_scores / sum_exp
        
        # Load V tile
        v = tl.load(v_ptrs + k_block * BLOCK_K * stride_vk, mask=(k_block * BLOCK_K + tl.arange(0, BLOCK_K) < seq_len)[:, None], other=0.0)
        
        # Compute attention output
        acc += tl.dot(probs, v, allow_tf32=False)
    
    # Write output
    out_ptrs = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh + m_block * stride_om
    tl.store(out_ptrs, acc, mask=(m_block * BLOCK_M + tl.arange(0, BLOCK_M) < seq_len)[:, None])

@triton.jit
def fused_matmul_softmax_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seq_len, head_dim,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    m_block = tl.program_id(2)
    
    # Initialize pointers for Q
    q_ptrs = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh + m_block * stride_qm
    
    # Initialize pointers for K and V
    k_ptrs = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_ptrs = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    
    # Initialize accumulator for attention scores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the sequence dimension
    for k_block in range(0, (seq_len + BLOCK_K - 1) // BLOCK_K):
        # Load Q, K, V tiles
        q = tl.load(q_ptrs + k_block * BLOCK_K * stride_qk, mask=(k_block * BLOCK_K + tl.arange(0, BLOCK_K) < seq_len)[None, :], other=0.0)
        
        k = tl.load(k_ptrs + k_block * BLOCK_K * stride_kk, mask=(k_block * BLOCK_K + tl.arange(0, BLOCK_K) < seq_len)[:, None], other=0.0)
        
        # Compute attention scores
        attn_scores = tl.dot(q, k.T, allow_tf32=False)
        attn_scores *= SCALE
        
        # Apply causal mask if needed (can be extended)
        # For now assuming no causal masking
        
        # Apply softmax
        max_val = tl.max(attn_scores, axis=1, keepdims=True)
        exp_scores = tl.exp(attn_scores - max_val)
        sum_exp = tl.sum(exp_scores, axis=1, keepdims=True)
        probs = exp_scores / sum_exp
        
        # Load V tile
        v = tl.load(v_ptrs + k_block * BLOCK_K * stride_vk, mask=(k_block * BLOCK_K + tl.arange(0, BLOCK_K) < seq_len)[:, None], other=0.0)
        
        # Compute attention output
        acc += tl.dot(probs, v, allow_tf32=False)
    
    # Write output
    out_ptrs = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh + m_block * stride_om
    tl.store(out_ptrs, acc, mask=(m_block * BLOCK_M + tl.arange(0, BLOCK_M) < seq_len)[:, None])

def triton_scaled_dot_product_attention(Q, K, V):
    """
    Triton implementation of scaled dot product attention
    """
    # Ensure tensors are contiguous and on GPU
    Q = Q.contiguous().to(torch.float32)
    K = K.contiguous().to(torch.float32)
    V = V.contiguous().to(torch.float32)
    
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Calculate scaling factor
    scale = 1.0 / math.sqrt(head_dim)
    
    # Create output tensor
    Out = torch.empty_like(Q)
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    # Grid configuration
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
        HEAD_DIM=head_dim,
        SCALE=scale
    )
    
    return Out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Use custom Triton kernel instead of PyTorch's scaled_dot_product_attention
        return triton_scaled_dot_product_attention(Q, K, V)