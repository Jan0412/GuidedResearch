import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,  # Input tensors
    Out,  # Output tensor
    stride_qz, stride_qh, stride_qm, stride_qk,  # Strides for Q
    stride_kz, stride_kh, stride_kn, stride_kk,  # Strides for K
    stride_vz, stride_vh, stride_vk, stride_vn,  # Strides for V
    stride_oz, stride_oh, stride_om, stride_on,  # Strides for output
    Z, H, N_CTX,  # Batch size, number of heads, sequence length
    D_HEAD: tl.constexpr,  # Head dimension
    BLOCK_M: tl.constexpr,  # Block size for Q rows
    BLOCK_N: tl.constexpr,  # Block size for K rows
    STAGE: tl.constexpr,  # Algorithm stage
):
    # Program IDs
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Offset pointers for batch and head
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Base pointers
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    
    # Block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    k_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(D_HEAD, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(D_HEAD, BLOCK_N),
        order=(0, 1)
    )
    v_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, D_HEAD),
        order=(1, 0)
    )
    
    # Initialize output accumulator
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    m = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    
    # Load Q block
    q = tl.load(Q_block_ptr)
    
    # Range of K and V blocks
    lo = 0
    hi = (start_m + 1) * BLOCK_M if STAGE == 2 else N_CTX
    
    # Loop over K and V blocks
    for start_n in range(lo, hi, BLOCK_N):
        # Load K block
        k = tl.load(k_block_ptr)
        
        # Compute Q @ K^T
        qk = tl.dot(q, k)
        
        # Apply scaling
        qk *= sm_scale
        
        # Compute mask for causal attention if needed
        if STAGE == 2:
            mask = start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] >= (start_n + tl.arange(0, BLOCK_N)[None, :])
            qk = tl.where(mask, qk, float("-inf"))
        
        # Compute online softmax
        m_new = tl.maximum(m, tl.max(qk, 1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(qk - m_new[:, None])
        l *= alpha
        l += tl.sum(p, 1)
        m = m_new
        
        # Update attention output
        acc *= alpha[:, None]
        p = p.to(tl.float16)  # Convert back to float16 for matrix multiplication
        
        # Load V block
        v = tl.load(v_block_ptr)
        
        # Compute P @ V
        p_transposed = tl.trans(p)
        acc = tl.dot(p_transposed, v, acc)
        
        # Update block pointers
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))
    
    # Normalize output
    acc = acc / l[:, None]
    
    # Write output
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))


def _get_block_size(N, max_block_size=64):
    """Get appropriate block size based on sequence length"""
    for block_size in [32, 64, 128, 256]:
        if N <= block_size:
            return block_size
    return max_block_size


def triton_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of scaled dot-product attention.
    This replaces torch.nn.functional.scaled_dot_product_attention.
    """
    # Ensure tensors are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Compute scale factor
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Prepare output tensor
    Out = torch.empty_like(Q)
    
    # Configure kernel parameters
    BLOCK_M = min(64, seq_len)  # Block size for Q rows
    BLOCK_N = min(64, seq_len)  # Block size for K rows
    
    # Grid configuration: (num_blocks_M, batch_size * num_heads)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)
    
    # Launch kernel
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        batch_size, num_heads, seq_len,
        D_HEAD=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=2  # Use causal attention stage
    )
    
    return Out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_attention(Q, K, V)