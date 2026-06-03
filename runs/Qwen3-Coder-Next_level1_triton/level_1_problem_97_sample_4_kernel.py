import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, 
    # Strides
    q_stride_b, q_stride_h, q_stride_s, q_stride_d,
    k_stride_b, k_stride_h, k_stride_s, k_stride_d,
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    # Dimensions
    batch_size, num_heads, seq_len, head_dim,
    # Scaling factor
    scale,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # Block size for Q rows
    BLOCK_SIZE_N: tl.constexpr,  # Block size for K/V rows
    BLOCK_SIZE_D: tl.constexpr,  # Block size for head dimension
    # Flags
    IS_CAUSAL: tl.constexpr,
):
    # Program IDs
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_id_m = tl.program_id(2)  # Block of Q rows
    
    # Offsets for batch and head
    q_offset = batch_id * q_stride_b + head_id * q_stride_h
    k_offset = batch_id * k_stride_b + head_id * k_stride_h
    v_offset = batch_id * v_stride_b + head_id * v_stride_h
    out_offset = batch_id * out_stride_b + head_id * out_stride_h
    
    # Initialize output accumulator
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_D], dtype=tl.float32)
    # Initialize max and sum for numerical stability in softmax
    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Loop over K/V columns (seq_len)
    for block_id_n in range(0, seq_len, BLOCK_SIZE_N):
        # Compute actual block end (might be smaller than BLOCK_SIZE_N)
        block_end_n = tl.minimum(block_id_n + BLOCK_SIZE_N, seq_len)
        
        # Compute Q @ K^T for this block
        # Load Q block: [BLOCK_SIZE_M, head_dim]
        q_start_m = block_id_m * BLOCK_SIZE_M
        q_end_m = tl.minimum(q_start_m + BLOCK_SIZE_M, seq_len)
        q_block_size = q_end_m - q_start_m
        
        # Create offsets for Q block
        q_row_offsets = q_start_m + tl.arange(0, BLOCK_SIZE_M)
        q_col_offsets = tl.arange(0, BLOCK_SIZE_D)
        q_mask = q_row_offsets[:, None] < q_end_m
        
        # Load Q block
        Q_block = tl.load(
            Q_ptr + q_offset + q_row_offsets[:, None] * q_stride_s + q_col_offsets[None, :] * q_stride_d,
            mask=q_mask,
            other=0.0
        )
        
        # Initialize accumulator for attention scores
        acc_scores = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        
        # Load K block: [BLOCK_SIZE_N, head_dim]
        k_row_offsets = block_id_n + tl.arange(0, BLOCK_SIZE_N)
        k_mask = k_row_offsets[:, None] < block_end_n
        
        K_block = tl.load(
            K_ptr + k_offset + k_row_offsets[:, None] * k_stride_s + q_col_offsets[None, :] * k_stride_d,
            mask=k_mask,
            other=0.0
        )
        
        # Compute Q @ K^T: [BLOCK_SIZE_M, BLOCK_SIZE_N]
        # Transpose K for efficient memory access
        acc_scores += tl.dot(Q_block, K_block.T)
        
        # Apply scaling
        acc_scores = acc_scores * scale
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            causal_mask = q_row_offsets[:, None] >= k_row_offsets[None, :]
            acc_scores = tl.where(causal_mask, acc_scores, float("-inf"))
        
        # Compute softmax with numerical stability
        # Find max per row
        m_ij = tl.max(acc_scores, 1)
        # Update running max and sum
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        l_i = l_i * alpha + tl.exp(m_ij - m_i_new)
        m_i = m_i_new
        
        # Softmax
        acc_scores = tl.exp(acc_scores - m_i_new[:, None])
        
        # Update output accumulator
        acc = acc * alpha[:, None]
        
        # Load V block: [BLOCK_SIZE_N, head_dim]
        v_block = tl.load(
            V_ptr + v_offset + k_row_offsets[:, None] * v_stride_s + q_col_offsets[None, :] * v_stride_d,
            mask=k_mask,
            other=0.0
        )
        
        # Compute softmax @ V
        acc += tl.dot(acc_scores, v_block)
    
    # Normalize output by softmax sum
    acc = acc / l_i[:, None]
    
    # Write output
    out_row_offsets = q_start_m + tl.arange(0, BLOCK_SIZE_M)
    out_col_offsets = tl.arange(0, BLOCK_SIZE_D)
    out_mask = out_row_offsets[:, None] < seq_len
    
    tl.store(
        Out_ptr + out_offset + out_row_offsets[:, None] * out_stride_s + out_col_offsets[None, :] * out_stride_d,
        acc,
        mask=out_mask
    )


def triton_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
    """
    Triton implementation of scaled dot-product attention.
    
    Args:
        Q: [batch_size, num_heads, seq_len, head_dim]
        K: [batch_size, num_heads, seq_len, head_dim]
        V: [batch_size, num_heads, seq_len, head_dim]
        is_causal: whether to apply causal masking
    
    Returns:
        Output tensor with same shape as input tensors
    """
    # Ensure contiguous tensors
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Output tensor
    Out = torch.empty_like(Q)
    
    # Compute scaling factor (1/sqrt(head_dim))
    scale = 1.0 / math.sqrt(head_dim)
    
    # Define block sizes - tuned for A100/A10
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_D = 64  # Should match head_dim or be a divisor
    
    # Ensure head_dim is divisible by BLOCK_SIZE_D for optimal performance
    # For simplicity, we assume head_dim == 1024 fits in multiple blocks or use smaller blocks
    
    # Adjust block size if head_dim is not divisible
    if head_dim % BLOCK_SIZE_D != 0:
        BLOCK_SIZE_D = 32  # Use smaller block size that works for any head_dim
    
    # Grid: [batch_size, num_heads, seq_len // BLOCK_SIZE_M]
    grid = (batch_size, num_heads, (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    
    # Launch kernel
    attention_kernel[grid](
        Q, K, V, Out,
        # Strides
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        # Dimensions
        batch_size, num_heads, seq_len, head_dim,
        # Scaling
        scale,
        # Block sizes
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        # Flags
        IS_CAUSAL=is_causal,
    )
    
    return Out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Use our optimized Triton attention implementation
        return triton_attention(Q, K, V, is_causal=False)