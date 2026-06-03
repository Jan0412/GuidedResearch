import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,  # Inputs
    Out,  # Output
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,  # Dimensions
    D_HEAD: tl.constexpr,  # Embedding dimension per head
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get program IDs
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Offsets for Z, H
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Pointer offsets
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
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(D_HEAD, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(D_HEAD, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, D_HEAD),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    
    # Load Q
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
    # Loop over blocks of K and V
    lo = 0
    hi = (start_m + 1) * BLOCK_M
    
    # Compute attention scores and accumulate
    for start_n in range(lo, hi, BLOCK_N):
        # Load K and V
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Compute Q @ K^T
        qk = tl.dot(q, k)
        qk = qk * sm_scale
        
        # Apply causal mask (for causal attention)
        # This is not strictly needed for non-causal but won't hurt
        mask = start_n + tl.arange(0, BLOCK_N)
        qk = tl.where(mask[None, :] < hi, qk, float("-inf"))
        
        # Compute online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp(m_i - m_ij)
        l_ij = tl.math.exp(qk - m_ij[None, :])
        
        # Update l_i and m_i
        l_i = p * l_i + tl.sum(l_ij, 1)
        m_i = m_ij
        
        # Update accumulator
        acc = acc * p[:, None]
        acc = acc + tl.dot(l_ij.to(tl.float16), v)
        
        # Advance K and V pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # Normalize output
    acc = acc / l_i[:, None]
    
    # Store output
    tl.store(O_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


@triton.jit
def _attn_bwd_preprocess_kernel(
    O, DO,  # Output and output gradients
    Delta,  # Precomputed delta for backprop
    stride_oz, stride_oh, stride_om, stride_on,
    stride_doz, stride_doh, stride_dom, stride_don,
    Z, H, N_CTX,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Get program IDs
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Offsets
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Pointer offsets
    o_offset = off_z * stride_oz + off_h * stride_oh
    do_offset = off_z * stride_doz + off_h * stride_doh
    
    # Block pointers
    O_block_ptr = tl.make_block_ptr(
        base=O + o_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_dom, stride_don),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    
    # Load O and DO
    o = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
    # Compute delta = sum(O * DO, axis=1)
    delta = tl.sum(o * do, axis=1)
    
    # Store delta
    tl.store(Delta + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M), delta, mask=tl.arange(0, BLOCK_M) < N_CTX)


@triton.jit
def _attn_bwd_kernel(
    Q, K, V, sm_scale,
    DO, DQ, DK, DV,
    Delta,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_doz, stride_doh, stride_dom, stride_don,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk, stride_dvn,
    Z, H, N_CTX,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # This is a simplified backward kernel - in practice, you'd need
    # the full implementation for training. For inference-only use,
    # we skip the backward kernel.
    pass


def scaled_dot_product_attention_triton(Q, K, V, sm_scale=None):
    """
    Triton implementation of scaled dot-product attention.
    Q, K, V: (batch_size, num_heads, seq_len, head_dim)
    """
    # Ensure inputs are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Set scale
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Output tensor
    Out = torch.empty_like(Q)
    
    # Grid configuration
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Grid: (num_blocks_m, batch_size * num_heads)
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
        num_warps=4,
        num_stages=2,
    )
    
    return Out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Use custom Triton implementation of scaled dot-product attention
        return scaled_dot_product_attention_triton(Q, K, V)