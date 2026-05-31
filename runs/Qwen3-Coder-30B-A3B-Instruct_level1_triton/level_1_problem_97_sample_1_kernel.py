import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def scaled_dot_product_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seq_len, head_dim,
    batch_size, num_heads,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_MASK: tl.constexpr
):
    # Get the block ID for the current program
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Calculate the starting positions for this block
    m_start = tl.program_id(2) * BLOCK_SIZE_M
    n_start = tl.program_id(3) * BLOCK_SIZE_N
    
    # Initialize accumulator for attention scores
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (split into chunks)
    for k_start in range(0, seq_len, BLOCK_SIZE_K):
        # Load Q, K, V tiles
        q = tl.load(Q_ptr + 
                   batch_id * num_heads * seq_len * head_dim +
                   head_id * seq_len * head_dim +
                   m_start * head_dim +
                   tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim +
                   tl.arange(0, BLOCK_SIZE_K)[None, :])
        
        k = tl.load(K_ptr + 
                   batch_id * num_heads * seq_len * head_dim +
                   head_id * seq_len * head_dim +
                   k_start * head_dim +
                   tl.arange(0, BLOCK_SIZE_K)[:, None] * head_dim +
                   tl.arange(0, BLOCK_SIZE_N)[None, :])
        
        v = tl.load(V_ptr + 
                   batch_id * num_heads * seq_len * head_dim +
                   head_id * seq_len * head_dim +
                   k_start * head_dim +
                   tl.arange(0, BLOCK_SIZE_K)[:, None] * head_dim +
                   tl.arange(0, BLOCK_SIZE_N)[None, :])
        
        # Compute attention scores (Q @ K^T)
        qk = tl.dot(q, k.T)
        
        # Scale the attention scores
        qk *= 1.0 / math.sqrt(head_dim)
        
        # Apply causal mask if needed
        if USE_MASK:
            # Create mask for causal attention
            mask = tl.arange(0, BLOCK_SIZE_M)[:, None] >= tl.arange(0, BLOCK_SIZE_N)[None, :]
            qk = tl.where(mask, qk, float('-inf'))
        
        # Apply softmax
        qk = qk - tl.max(qk, axis=1, keepdims=True)
        exp_qk = tl.exp(qk)
        sum_exp_qk = tl.sum(exp_qk, axis=1, keepdims=True)
        softmax_qk = exp_qk / sum_exp_qk
        
        # Accumulate the weighted values
        acc += tl.dot(softmax_qk, v)
    
    # Write back the result
    out_ptr = Out_ptr + batch_id * num_heads * seq_len * head_dim + \
              head_id * seq_len * head_dim + \
              m_start * head_dim + \
              tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim + \
              tl.arange(0, BLOCK_SIZE_N)[None, :]
    
    tl.store(out_ptr, acc, mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_len) & 
                                           (tl.arange(0, BLOCK_SIZE_N)[None, :] < seq_len))

# Simplified fused implementation for better performance
@triton.jit
def fused_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seq_len, head_dim,
    batch_size, num_heads,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    m_start = tl.program_id(2) * BLOCK_SIZE_M
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension in chunks
    for k_start in range(0, seq_len, BLOCK_SIZE_K):
        # Load tiles from memory
        q = tl.load(Q_ptr + 
                   batch_id * num_heads * seq_len * head_dim +
                   head_id * seq_len * head_dim +
                   m_start * head_dim +
                   tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim +
                   tl.arange(0, BLOCK_SIZE_K)[None, :])
        
        k = tl.load(K_ptr + 
                   batch_id * num_heads * seq_len * head_dim +
                   head_id * seq_len * head_dim +
                   k_start * head_dim +
                   tl.arange(0, BLOCK_SIZE_K)[:, None] * head_dim +
                   tl.arange(0, BLOCK_SIZE_N)[None, :])
        
        v = tl.load(V_ptr + 
                   batch_id * num_heads * seq_len * head_dim +
                   head_id * seq_len * head_dim +
                   k_start * head_dim +
                   tl.arange(0, BLOCK_SIZE_K)[:, None] * head_dim +
                   tl.arange(0, BLOCK_SIZE_N)[None, :])
        
        # Compute attention scores (Q @ K^T)
        qk = tl.dot(q, k.T)
        
        # Scale the attention scores
        qk *= SCALE
        
        # Apply causal mask
        mask = tl.arange(0, BLOCK_SIZE_M)[:, None] >= tl.arange(0, BLOCK_SIZE_N)[None, :]
        qk = tl.where(mask, qk, float('-inf'))
        
        # Apply softmax
        qk = qk - tl.max(qk, axis=1, keepdims=True)
        exp_qk = tl.exp(qk)
        sum_exp_qk = tl.sum(exp_qk, axis=1, keepdims=True)
        softmax_qk = exp_qk / sum_exp_qk
        
        # Accumulate the weighted values
        acc += tl.dot(softmax_qk, v)
    
    # Store results
    out_ptr = Out_ptr + batch_id * num_heads * seq_len * head_dim + \
              head_id * seq_len * head_dim + \
              m_start * head_dim + \
              tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim + \
              tl.arange(0, BLOCK_SIZE_N)[None, :]
    
    tl.store(out_ptr, acc)

def triton_scaled_dot_product_attention(Q, K, V):
    """
    Triton implementation of scaled dot product attention
    """
    assert Q.dtype == torch.float16, "Expected float16 input"
    assert K.dtype == torch.float16, "Expected float16 input"
    assert V.dtype == torch.float16, "Expected float16 input"
    
    # Ensure inputs are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Output tensor
    Out = torch.empty_like(Q)
    
    # Define constants
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    SCALE = 1.0 / math.sqrt(head_dim)
    
    # Grid configuration
    grid = (
        batch_size,
        num_heads,
        (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )
    
    # Launch kernel
    fused_attention_kernel[grid](
        Q, K, V, Out,
        seq_len, head_dim,
        batch_size, num_heads,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SCALE=SCALE
    )
    
    return Out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_scaled_dot_product_attention(Q, K, V)