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
    USE_F32: tl.constexpr
):
    # Get the program ID
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Initialize pointers to Q, K, V, and output
    q_ptr = Q_ptr + batch_id * num_heads * seq_len * head_dim + head_id * seq_len * head_dim
    k_ptr = K_ptr + batch_id * num_heads * seq_len * head_dim + head_id * seq_len * head_dim
    v_ptr = V_ptr + batch_id * num_heads * seq_len * head_dim + head_id * seq_len * head_dim
    out_ptr = Out_ptr + batch_id * num_heads * seq_len * head_dim + head_id * seq_len * head_dim
    
    # Create block-level matrices
    m = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    d = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Loop over tiles
    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        # Load K and V tiles
        k_tile = tl.load(k_ptr + start_n * head_dim + tl.arange(0, BLOCK_SIZE_N)[:, None] * head_dim + tl.arange(0, BLOCK_SIZE_K)[None, :], 
                        mask=(tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - start_n) & 
                              (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim))
        
        v_tile = tl.load(v_ptr + start_n * head_dim + tl.arange(0, BLOCK_SIZE_N)[:, None] * head_dim + tl.arange(0, BLOCK_SIZE_K)[None, :], 
                        mask=(tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - start_n) & 
                              (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim))
        
        # Compute attention scores
        qk = tl.dot(tl.load(q_ptr + tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim + tl.arange(0, BLOCK_SIZE_K)[None, :], 
                           mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_len) & 
                                 (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim)),
                   k_tile, trans_b=True) / math.sqrt(head_dim)
        
        # Apply softmax
        qk = qk - tl.max(qk, axis=1, keepdims=True)
        qk = tl.exp(qk)
        sum_qk = tl.sum(qk, axis=1, keepdims=True)
        qk = qk / sum_qk
        
        # Compute output tile
        out_tile = tl.dot(qk, v_tile)
        
        # Store output
        tl.store(out_ptr + tl.arange(0, BLOCK_SIZE_M)[:, None] * head_dim + tl.arange(0, BLOCK_SIZE_K)[None, :], 
                out_tile, mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_len) & 
                              (tl.arange(0, BLOCK_SIZE_K)[None, :] < head_dim))

def triton_scaled_dot_product_attention(Q, K, V):
    """Custom Triton implementation of scaled dot product attention"""
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Ensure tensors are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Allocate output tensor
    Out = torch.empty_like(Q)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid dimensions
    grid = (
        batch_size,
        num_heads,
        1
    )
    
    # Launch kernel
    scaled_dot_product_attention_kernel[grid](
        Q, K, V, Out,
        seq_len, head_dim,
        batch_size, num_heads,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        USE_F32=False
    )
    
    return Out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_scaled_dot_product_attention(Q, K, V)