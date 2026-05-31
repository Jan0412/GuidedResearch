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
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PRECISION: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    m_id = tl.program_id(2)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load Q fragment
    q_offset = batch_id * stride_qb + head_id * stride_qh + m_id * stride_qm
    q_block = tl.load(Q_ptr + q_offset + tl.arange(0, BLOCK_K), mask=tl.arange(0, BLOCK_K) < head_dim, other=0.0)
    
    # Loop over K,V blocks
    for n_id in range(0, (seq_len + BLOCK_N - 1) // BLOCK_N):
        # Load K fragment
        k_offset = batch_id * stride_kb + head_id * stride_kh + n_id * stride_km
        k_block = tl.load(K_ptr + k_offset + tl.arange(0, BLOCK_K), mask=tl.arange(0, BLOCK_K) < head_dim, other=0.0)
        
        # Compute attention scores
        score = tl.sum(q_block[:, None] * k_block[None, :], axis=0)
        score = score / tl.sqrt(head_dim * 1.0)
        
        # Apply softmax approximation (simplified version)
        max_score = tl.max(score, axis=0)
        score = score - max_score
        exp_score = tl.exp(score)
        sum_exp_score = tl.sum(exp_score, axis=0)
        softmax_score = exp_score / sum_exp_score
        
        # Load V fragment
        v_offset = batch_id * stride_vb + head_id * stride_vh + n_id * stride_vm
        v_block = tl.load(V_ptr + v_offset + tl.arange(0, BLOCK_K), mask=tl.arange(0, BLOCK_K) < head_dim, other=0.0)
        
        # Accumulate weighted V
        acc += softmax_score[:, None] * v_block[None, :]
    
    # Write output
    out_offset = batch_id * stride_ob + head_id * stride_oh + m_id * stride_om
    tl.store(Out_ptr + out_offset + tl.arange(0, BLOCK_N), acc, mask=tl.arange(0, BLOCK_N) < seq_len)

# Optimized implementation using proper fused attention kernel
@triton.jit
def fused_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch_size, num_heads, seq_len, head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    m_id = tl.program_id(2)
    
    # Load Q row
    q_row = tl.load(Q_ptr + 
                    batch_id * stride_qb + 
                    head_id * stride_qh + 
                    m_id * stride_qm + 
                    tl.arange(0, BLOCK_K), 
                    mask=tl.arange(0, BLOCK_K) < head_dim, 
                    other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over blocks of K and V
    for n_id in range(0, (seq_len + BLOCK_N - 1) // BLOCK_N):
        # Load K block
        k_block = tl.load(K_ptr + 
                          batch_id * stride_kb + 
                          head_id * stride_kh + 
                          n_id * stride_km + 
                          tl.arange(0, BLOCK_K)[:, None], 
                          mask=(tl.arange(0, BLOCK_K)[:, None] < head_dim) & 
                                (tl.arange(0, BLOCK_N)[None, :] < seq_len - n_id * BLOCK_N),
                          other=0.0)
        
        # Compute attention scores: Q @ K^T
        scores = tl.dot(q_row[None, :], k_block) * scale
        
        # Apply softmax (approximation)
        max_scores = tl.max(scores, axis=1, keepdims=True)
        scores = scores - max_scores
        exp_scores = tl.exp(scores)
        sum_exp_scores = tl.sum(exp_scores, axis=1, keepdims=True)
        softmax_scores = exp_scores / sum_exp_scores
        
        # Load V block
        v_block = tl.load(V_ptr + 
                          batch_id * stride_vb + 
                          head_id * stride_vh + 
                          n_id * stride_vm + 
                          tl.arange(0, BLOCK_K)[None, :], 
                          mask=(tl.arange(0, BLOCK_K)[None, :] < head_dim) & 
                                (tl.arange(0, BLOCK_N)[:, None] < seq_len - n_id * BLOCK_N),
                          other=0.0)
        
        # Accumulate: softmax_scores @ V
        acc += tl.dot(softmax_scores, v_block)
    
    # Write output
    out_row = Out_ptr + batch_id * stride_ob + head_id * stride_oh + m_id * stride_om
    tl.store(out_row + tl.arange(0, BLOCK_N), acc[0], mask=tl.arange(0, BLOCK_N) < seq_len)

def triton_scaled_dot_product_attention(Q, K, V):
    """Custom Triton implementation of scaled dot product attention"""
    # Ensure inputs are contiguous and on CUDA
    Q = Q.contiguous().to(torch.float32)
    K = K.contiguous().to(torch.float32)
    V = V.contiguous().to(torch.float32)
    
    # Extract dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Calculate scaling factor
    scale = 1.0 / math.sqrt(float(head_dim))
    
    # Prepare output tensor
    Out = torch.empty_like(Q)
    
    # Define block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = head_dim
    
    # Grid configuration
    grid = (
        batch_size,  # Batch dimension
        num_heads,   # Head dimension  
        (seq_len + BLOCK_M - 1) // BLOCK_M  # Sequence dimension
    )
    
    # Launch kernel
    fused_attention_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        batch_size, num_heads, seq_len, head_dim,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    
    return Out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_scaled_dot_product_attention(Q, K, V)