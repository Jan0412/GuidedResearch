import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_attention(
    Q, K, V,
    sm_scale,
    Out,
    q_stride_h, q_stride_m, q_stride_d,
    k_stride_h, k_stride_m, k_stride_d,
    v_stride_h, v_stride_m, v_stride_d,
    o_stride_h, o_stride_m, o_stride_d,
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # number of heads
    Nq: tl.constexpr,  # sequence length Q
    Nk: tl.constexpr,  # sequence length K
    D: tl.constexpr,  # head dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get program indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_m_idx = tl.program_id(2)

    # Offset pointers for batch and head
    q_offset = batch_idx * q_stride_h + head_idx * q_stride_m
    k_offset = batch_idx * k_stride_h + head_idx * k_stride_m
    v_offset = batch_idx * v_stride_h + head_idx * v_stride_m
    o_offset = batch_idx * o_stride_h + head_idx * o_stride_m

    # Initialize output accumulator
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Initialize max and sum for softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Compute softmax scale
    sm_scale = sm_scale

    # Loop over blocks of keys
    for block_n in range(0, tl.cdiv(Nk, BLOCK_N)):
        # Compute start index for key blocks
        start_n = block_n * BLOCK_N
        
        # Load Q block
        block_m_offsets = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = block_m_offsets < Nq
        
        q_ptrs = q_offset + block_m_offsets[:, None] * q_stride_m + tl.arange(0, D)[None, :] * q_stride_d
        q_ptrs = tl.max_contiguous(tl.multiple_of(q_ptrs, 16), D)
        
        # Load Q block
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        
        # Initialize accumulator for this Q block
        acc_block = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        
        # Compute dot products with K blocks
        for block_k in range(0, tl.cdiv(Nk, BLOCK_N)):
            start_k = block_k * BLOCK_N
            block_n_offsets = start_k + tl.arange(0, BLOCK_N)
            mask_n = block_n_offsets < Nk
            
            # Load K block (transposed)
            k_ptrs = k_offset + block_n_offsets[None, :] * k_stride_m + tl.arange(0, D)[:, None] * k_stride_d
            k_ptrs = tl.max_contiguous(tl.multiple_of(k_ptrs, 16), D)
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
            
            # Compute QK^T
            qk = tl.dot(q, k)
            
            # Apply scaling
            qk = qk * sm_scale
            
            # Load V block for later
            v_ptrs = v_offset + block_n_offsets[:, None] * v_stride_m + tl.arange(0, D)[None, :] * v_stride_d
            v_ptrs = tl.max_contiguous(tl.multiple_of(v_ptrs, 16), D)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            
            # Compute attention scores with masking
            qk = qk.to(tl.float32)
            
            # For causal attention, we would apply masking here
            # For now, assume standard attention
            
            # Compute softmax
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            
            # Update softmax statistics
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            
            # Update output accumulator
            acc_block = acc_block * alpha[:, None]
            
            # Compute p @ V
            pv = tl.dot(p.to(v.dtype), v)
            acc_block = acc_block + pv
            
        # Store final result for this Q block
        out_ptrs = o_offset + block_m_offsets[:, None] * o_stride_m + tl.arange(0, D)[None, :] * o_stride_d
        out_ptrs = tl.max_contiguous(tl.multiple_of(out_ptrs, 16), D)
        
        # Normalize output
        acc_block = acc_block / l_i[:, None]
        
        # Store output
        tl.store(out_ptrs, acc_block.to(Out.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _fwd_kernel_flash_attention_v2(
    Q, K, V,
    sm_scale,
    Out,
    q_stride_h, q_stride_m, q_stride_d,
    k_stride_h, k_stride_m, k_stride_d,
    v_stride_h, v_stride_m, v_stride_d,
    o_stride_h, o_stride_m, o_stride_d,
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # number of heads
    Nq: tl.constexpr,  # sequence length Q
    Nk: tl.constexpr,  # sequence length K
    D: tl.constexpr,  # head dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get program indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_m_idx = tl.program_id(2)

    # Offset pointers for batch and head
    q_offset = batch_idx * q_stride_h + head_idx * q_stride_m
    k_offset = batch_idx * k_stride_h + head_idx * k_stride_m
    v_offset = batch_idx * v_stride_h + head_idx * v_stride_m
    o_offset = batch_idx * o_stride_h + head_idx * o_stride_m

    # Initialize output accumulator
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Initialize max and sum for softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Compute softmax scale
    sm_scale = sm_scale

    # Loop over blocks of keys
    for block_n in range(0, tl.cdiv(Nk, BLOCK_N)):
        # Compute start index for key blocks
        start_n = block_n * BLOCK_N
        
        # Load Q block
        block_m_offsets = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = block_m_offsets < Nq
        
        q_ptrs = q_offset + block_m_offsets[:, None] * q_stride_m + tl.arange(0, D)[None, :] * q_stride_d
        q_ptrs = tl.max_contiguous(tl.multiple_of(q_ptrs, 16), D)
        
        # Load Q block
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        
        # Initialize accumulator for this Q block
        acc_block = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        
        # Compute dot products with K blocks
        for block_k in range(0, tl.cdiv(Nk, BLOCK_N)):
            start_k = block_k * BLOCK_N
            block_n_offsets = start_k + tl.arange(0, BLOCK_N)
            mask_n = block_n_offsets < Nk
            
            # Load K block (transposed)
            k_ptrs = k_offset + block_n_offsets[None, :] * k_stride_m + tl.arange(0, D)[:, None] * k_stride_d
            k_ptrs = tl.max_contiguous(tl.multiple_of(k_ptrs, 16), D)
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
            
            # Compute QK^T
            qk = tl.dot(q, k)
            
            # Apply scaling
            qk = qk * sm_scale
            
            # Load V block for later
            v_ptrs = v_offset + block_n_offsets[:, None] * v_stride_m + tl.arange(0, D)[None, :] * v_stride_d
            v_ptrs = tl.max_contiguous(tl.multiple_of(v_ptrs, 16), D)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            
            # Compute attention scores with masking
            qk = qk.to(tl.float32)
            
            # For causal attention, we would apply masking here
            # For now, assume standard attention
            
            # Compute softmax
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            
            # Update softmax statistics
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            
            # Update output accumulator
            acc_block = acc_block * alpha[:, None]
            
            # Compute p @ V
            pv = tl.dot(p.to(v.dtype), v)
            acc_block = acc_block + pv
            
        # Store final result for this Q block
        out_ptrs = o_offset + block_m_offsets[:, None] * o_stride_m + tl.arange(0, D)[None, :] * o_stride_d
        out_ptrs = tl.max_contiguous(tl.multiple_of(out_ptrs, 16), D)
        
        # Normalize output
        acc_block = acc_block / l_i[:, None]
        
        # Store output
        tl.store(out_ptrs, acc_block.to(Out.dtype.element_ty), mask=mask_m[:, None])


def triton_scaled_dot_product_attention(Q, K, V):
    """
    Triton implementation of scaled dot product attention.
    Optimized for FP32 precision but accepts FP16 inputs.
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Tensors must be on CUDA."
    
    # Ensure contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Get dimensions
    B, H, Nq, D = Q.shape
    _, _, Nk, _ = K.shape
    
    # Calculate softmax scale
    sm_scale = 1.0 / (D ** 0.5)
    
    # Create output tensor
    Out = torch.empty_like(Q)
    
    # Calculate strides
    q_stride_h, q_stride_m, q_stride_d = Q.stride()[0], Q.stride()[1], Q.stride()[2]
    k_stride_h, k_stride_m, k_stride_d = K.stride()[0], K.stride()[1], K.stride()[2]
    v_stride_h, v_stride_m, v_stride_d = V.stride()[0], V.stride()[1], V.stride()[2]
    o_stride_h, o_stride_m, o_stride_d = Out.stride()[0], Out.stride()[1], Out.stride()[2]
    
    # Configure block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Calculate grid dimensions
    grid = (B, H, triton.cdiv(Nq, BLOCK_M))
    
    # Launch kernel
    _fwd_kernel_flash_attention[grid](
        Q, K, V,
        sm_scale,
        Out,
        q_stride_h, q_stride_m, q_stride_d,
        k_stride_h, k_stride_m, k_stride_d,
        v_stride_h, v_stride_m, v_stride_d,
        o_stride_h, o_stride_m, o_stride_d,
        B=B, H=H, Nq=Nq, Nk=Nk, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )
    
    return Out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_scaled_dot_product_attention(Q, K, V)