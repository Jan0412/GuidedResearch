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
    USE_FUSED: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Calculate the starting positions for this program
    q_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim
    k_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim
    v_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim
    out_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim
    
    # Initialize accumulator for attention scores
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over blocks of K and V
    for k_start in range(0, seq_len, BLOCK_SIZE_K):
        # Load Q, K, V tiles
        q = tl.load(Q_ptr + q_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim + 
                   tl.arange(0, BLOCK_SIZE_K)[None, :], 
                   mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_len - k_start) &
                         (tl.arange(0, BLOCK_SIZE_K)[None, :] < seq_len - k_start),
                   other=0.0)
        
        k = tl.load(K_ptr + k_offset + tl.arange(0, BLOCK_SIZE_K)[:, None] * head_dim + 
                   tl.arange(0, BLOCK_SIZE_N)[None, :], 
                   mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < seq_len - k_start) &
                         (tl.arange(0, BLOCK_SIZE_N)[None, :] < seq_len - k_start),
                   other=0.0)
        
        # Compute attention scores (Q @ K^T)
        qk = tl.dot(q, k, trans_b=True)
        
        # Scale by sqrt(d_k)
        scale = 1.0 / math.sqrt(head_dim)
        qk = qk * scale
        
        # Apply causal mask if needed (can be extended)
        # For simplicity, we're not implementing causal masking here
        
        # Apply softmax using log-sum-exp trick for numerical stability
        if USE_FUSED:
            # Apply softmax directly
            qk = tl.softmax(qk)
        
        # Load V tile
        v = tl.load(V_ptr + v_offset + tl.arange(0, BLOCK_SIZE_K)[:, None] * head_dim + 
                   tl.arange(0, BLOCK_SIZE_N)[None, :], 
                   mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < seq_len - k_start) &
                         (tl.arange(0, BLOCK_SIZE_N)[None, :] < seq_len - k_start),
                   other=0.0)
        
        # Compute attention output
        acc += tl.dot(qk, v)
    
    # Write output
    out = acc.to(tl.float16)
    tl.store(Out_ptr + out_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim + 
             tl.arange(0, BLOCK_SIZE_N)[None, :], out,
             mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_len) &
                   (tl.arange(0, BLOCK_SIZE_N)[None, :] < seq_len))

# More optimized version using fused operations
@triton.jit
def fused_scaled_dot_product_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seq_len, head_dim,
    batch_size, num_heads,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Calculate the starting positions for this program
    q_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim
    k_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim
    v_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim
    out_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim
    
    # Initialize accumulator for attention scores
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over blocks of K and V
    for k_start in range(0, seq_len, BLOCK_SIZE_K):
        # Load Q, K, V tiles
        q = tl.load(Q_ptr + q_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim + 
                   tl.arange(0, BLOCK_SIZE_K)[None, :], 
                   mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_len - k_start) &
                         (tl.arange(0, BLOCK_SIZE_K)[None, :] < seq_len - k_start),
                   other=0.0)
        
        k = tl.load(K_ptr + k_offset + tl.arange(0, BLOCK_SIZE_K)[:, None] * head_dim + 
                   tl.arange(0, BLOCK_SIZE_N)[None, :], 
                   mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < seq_len - k_start) &
                         (tl.arange(0, BLOCK_SIZE_N)[None, :] < seq_len - k_start),
                   other=0.0)
        
        # Compute attention scores (Q @ K^T)
        qk = tl.dot(q, k, trans_b=True)
        
        # Scale by sqrt(d_k)
        scale = 1.0 / math.sqrt(head_dim)
        qk = qk * scale
        
        # Apply softmax using log-sum-exp trick for numerical stability
        # Subtract max for numerical stability
        qk_max = tl.max(qk, axis=1, keepdims=True)
        qk = qk - qk_max
        # Compute exp
        qk = tl.exp(qk)
        # Compute sum for normalization
        qk_sum = tl.sum(qk, axis=1, keepdims=True)
        # Normalize
        qk = qk / qk_sum
        
        # Load V tile
        v = tl.load(V_ptr + v_offset + tl.arange(0, BLOCK_SIZE_K)[:, None] * head_dim + 
                   tl.arange(0, BLOCK_SIZE_N)[None, :], 
                   mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < seq_len - k_start) &
                         (tl.arange(0, BLOCK_SIZE_N)[None, :] < seq_len - k_start),
                   other=0.0)
        
        # Compute attention output
        acc += tl.dot(qk, v)
    
    # Write output
    out = acc.to(tl.float16)
    tl.store(Out_ptr + out_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim + 
             tl.arange(0, BLOCK_SIZE_N)[None, :], out,
             mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_len) &
                   (tl.arange(0, BLOCK_SIZE_N)[None, :] < seq_len))

def triton_scaled_dot_product_attention(Q, K, V):
    """Custom Triton implementation of scaled dot-product attention"""
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "All tensors must be on CUDA"
    assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16, "All tensors must be float16"
    
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Prepare output tensor
    Out = torch.empty_like(Q)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid configuration
    grid = (batch_size, num_heads)
    
    # Launch kernel
    fused_scaled_dot_product_attention_kernel[grid](
        Q, K, V, Out,
        seq_len, head_dim,
        batch_size, num_heads,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return Out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = triton_scaled_dot_product_attention(Q, K, V)
        return out