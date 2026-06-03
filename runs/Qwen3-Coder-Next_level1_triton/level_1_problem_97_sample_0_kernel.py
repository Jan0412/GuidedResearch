import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,  # pointers to inputs
    Out,  # pointer to output
    stride_qz, stride_qh, stride_qm, stride_qk,  # strides for Q
    stride_kz, stride_kh, stride_kn, stride_kk,  # strides for K
    stride_vz, stride_vh, stride_vk, stride_vn,  # strides for V
    stride_oz, stride_oh, stride_om, stride_on,  # strides for output
    Z, H, N_CTX, D_HEAD,  # dimensions
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    # Calculate program IDs
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Offsets for batch and head
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Pointers to input/output
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    
    # Block pointers
    q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    k_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(D_HEAD, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    
    v_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # Initialize output accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    
    # Load Q block
    q = tl.load(q_block_ptr)
    
    # Loop over K and V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K and V blocks
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)
        
        # Compute QK^T
        qk = tl.dot(q, k)
        qk = qk * sm_scale
        
        # Apply causal mask if needed (not in this case, but keep for generality)
        # For causal attention: mask = tl.where(tl.arange(0, BLOCK_M)[:, None] >= tl.arange(0, BLOCK_N)[None, :], 0, -float("inf"))
        # qk = qk + mask
        
        # Compute online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1) + tl.exp(m_i - m_ij)
        
        # Update attention output
        acc = acc * (l_i / l_ij)[:, None]
        acc = acc + tl.dot(p.to(tl.float16), v)
        
        # Update statistics for next iteration
        m_i = m_ij
        l_i = l_ij
        
        # Move block pointers forward
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))
    
    # Normalize output
    acc = acc / l_i[:, None]
    
    # Write output
    o_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(o_block_ptr, acc.to(tl.float16))


def attention_triton(Q, K, V, sm_scale=None):
    """
    Triton implementation of scaled dot-product attention.
    
    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        V: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        sm_scale: Scaling factor for attention (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    if sm_scale is None:
        sm_scale = 1.0 / (Q.shape[-1] ** 0.5)
    
    # Ensure inputs are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Allocate output
    Out = torch.empty_like(Q)
    
    # Define block sizes for Triton kernel
    BLOCK_M = 64  # Block size for sequence dimension
    BLOCK_N = 64  # Block size for key sequence dimension
    BLOCK_DMODEL = head_dim  # Head dimension
    
    # Calculate grid dimensions
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)
    
    # Calculate strides
    stride_qz, stride_qh, stride_qm, stride_qk = Q.stride()
    stride_kz, stride_kh, stride_kn, stride_kk = K.stride()
    stride_vz, stride_vh, stride_vk, stride_vn = V.stride()
    stride_oz, stride_oh, stride_om, stride_on = Out.stride()
    
    # Launch kernel
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return Out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Use our custom Triton attention implementation
        return attention_triton(Q, K, V)