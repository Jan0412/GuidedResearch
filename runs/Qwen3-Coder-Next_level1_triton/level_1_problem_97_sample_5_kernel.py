import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scaled_dot_attention_kernel(
    Q, K, V, Out,
    scale,
    num_heads, seq_len, embed_dim,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_km, stride_kd,
    stride_vh, stride_vm, stride_vd,
    stride_oh, stride_om, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Program IDs for head, sequence position in Q
    head_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # Initialize max and sum for online softmax
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Q block: [BLOCK_M, BLOCK_D]
    q_offsets = (
        head_id * stride_qh + 
        seq_id * stride_qm + 
        tl.arange(0, BLOCK_M)[:, None] * stride_qm +
        tl.arange(0, BLOCK_D)[None, :] * stride_qd
    )
    q_mask = (tl.arange(0, BLOCK_M)[:, None] < seq_len) & (tl.arange(0, BLOCK_D)[None, :] < embed_dim)
    q = tl.load(Q + q_offsets, mask=q_mask, other=0.0).to(tl.float32)
    
    # Accumulator over K/V sequence positions
    for start_n in range(0, seq_len, BLOCK_N):
        # K block: [BLOCK_N, BLOCK_D]
        k_offsets = (
            head_id * stride_kh +
            (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_km +
            tl.arange(0, BLOCK_D)[:, None] * stride_kd
        )
        k_mask = ((start_n + tl.arange(0, BLOCK_N))[None, :] < seq_len) & (tl.arange(0, BLOCK_D)[:, None] < embed_dim)
        k = tl.load(K + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        
        # Compute QK^T: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k)
        qk = qk * scale
        
        # Online softmax: compute max for numerical stability
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.exp(qk - m_ij[:, None])
        m_i = m_ij
        
        # V block: [BLOCK_N, BLOCK_D]
        v_offsets = (
            head_id * stride_vh +
            (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_vm +
            tl.arange(0, BLOCK_D)[None, :] * stride_vd
        )
        v_mask = ((start_n + tl.arange(0, BLOCK_N))[:, None] < seq_len) & (tl.arange(0, BLOCK_D)[None, :] < embed_dim)
        v = tl.load(V + v_offsets, mask=v_mask, other=0.0).to(tl.float32)
        
        # Compute attention weights * V
        p = tl.exp(qk - m_ij[:, None])
        acc = acc * alpha[:, None] + tl.dot(p, v)
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    out_offsets = (
        head_id * stride_oh +
        seq_id * stride_om +
        tl.arange(0, BLOCK_D)[None, :] * stride_od
    )
    out_mask = (tl.arange(0, BLOCK_D)[None, :] < embed_dim)
    tl.store(Out + out_offsets, acc.to(Out.dtype.element_ty), mask=out_mask)


def triton_scaled_dot_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of scaled dot-product attention.
    Q, K, V: [batch, num_heads, seq_len, embed_dim]
    Returns: [batch, num_heads, seq_len, embed_dim]
    """
    assert Q.shape == K.shape == V.shape, "Q, K, V must have same shape"
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Tensors must be on CUDA."
    
    batch_size, num_heads, seq_len, embed_dim = Q.shape
    
    # Ensure contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Prepare output tensor (same shape as inputs)
    Out = torch.empty_like(Q)
    
    # Strides for memory access
    stride_qh, stride_qm, stride_qd = Q.stride()[1], Q.stride()[2], Q.stride()[3]
    stride_kh, stride_km, stride_kd = K.stride()[1], K.stride()[2], K.stride()[3]
    stride_vh, stride_vm, stride_vd = V.stride()[1], V.stride()[2], V.stride()[3]
    stride_oh, stride_om, stride_od = Out.stride()[1], Out.stride()[2], Out.stride()[3]
    
    # Scale factor: 1/sqrt(embed_dim)
    scale = 1.0 / (embed_dim ** 0.5)
    
    # Block sizes for Triton (tunable parameters)
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = embed_dim  # Use full embedding dimension per block
    
    # Grid: [num_heads, seq_len]
    grid = (num_heads, seq_len)
    
    # Launch kernel
    scaled_dot_attention_kernel[grid](
        Q, K, V, Out,
        scale,
        num_heads, seq_len, embed_dim,
        stride_qh, stride_qm, stride_qd,
        stride_kh, stride_km, stride_kd,
        stride_vh, stride_vm, stride_vd,
        stride_oh, stride_om, stride_od,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    return Out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Use our optimized Triton implementation
        return triton_scaled_dot_attention(Q, K, V)