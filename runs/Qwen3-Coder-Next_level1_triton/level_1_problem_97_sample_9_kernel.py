import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _attn_forward_kernel(
    Q, K, V,
    sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kk, stride_kn,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, D_HEAD,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    # Indices for the current block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Number of blocks
    num_blocks_m = tl.cdiv(N_CTX, BLOCK_M)
    
    # Offsets for Q
    q_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_offsets < N_CTX
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, D_HEAD),
        strides=(stride_qm, stride_qk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    
    # Initialize accumulator for output
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    # For stages 1 and 3, we compute the attention scores
    if STAGE == 1:
        # Compute attention scores for the current block of Q with all K
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Compute scores with all K blocks
        max_score = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        sum_exp = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
        
        # Loop over K blocks
        for start_n in range(0, N_CTX, BLOCK_N):
            k_offsets = start_n + tl.arange(0, BLOCK_N)
            k_mask = k_offsets < N_CTX
            
            # Load K block
            K_block_ptr = tl.make_block_ptr(
                base=K,
                shape=(D_HEAD, N_CTX),
                strides=(stride_kk, stride_kn),
                offsets=(0, start_n),
                block_shape=(D_HEAD, BLOCK_N),
                order=(0, 1)
            )
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            
            # Compute QK^T
            attn_scores = tl.dot(q, k) * sm_scale
            
            # Apply causal mask if needed (for now assume no causal mask)
            # For full attention, we don't need masking
            
            # Compute online softmax
            curr_max = tl.max(attn_scores, axis=1)
            curr_max = tl.where(q_mask, curr_max, float("-inf"))
            max_score = tl.maximum(max_score, curr_max)
            
            # Compute exp scores
            attn_scores = attn_scores - max_score[:, None]
            attn_scores = tl.where(q_mask[:, None], attn_scores, float("-inf"))
            exp_scores = tl.exp(attn_scores)
            sum_exp = sum_exp + tl.sum(exp_scores, axis=1)
            
            # Load V block
            V_block_ptr = tl.make_block_ptr(
                base=V,
                shape=(N_CTX, D_HEAD),
                strides=(stride_vk, stride_vn),
                offsets=(start_n, 0),
                block_shape=(BLOCK_N, D_HEAD),
                order=(1, 0)
            )
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            
            # Accumulate weighted V
            weights = exp_scores.to(v.dtype)
            acc = acc + tl.dot(weights, v)
        
        # Normalize by sum_exp
        sum_exp = tl.where(q_mask, sum_exp, 0.0)
        acc = acc / sum_exp[:, None]
        
        # Store output
        Out_block_ptr = tl.make_block_ptr(
            base=Out,
            shape=(N_CTX, D_HEAD),
            strides=(stride_om, stride_on),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, D_HEAD),
            order=(1, 0)
        )
        tl.store(Out_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))
        
    elif STAGE == 2:
        # For causal attention, we need to handle the causal mask
        pass
    else:  # STAGE == 3
        # Alternative implementation using different blocking strategy
        pass

@triton.jit
def _attn_forward_kernel_fused(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kk, stride_kn,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, D_HEAD,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    # This is a more efficient implementation that fuses the operations
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(N_CTX, BLOCK_M)
    pid_m = pid // H
    pid_h = pid % H
    
    # Offsets for Q
    q_start = pid_m * BLOCK_M
    q_offsets = q_start + tl.arange(0, BLOCK_M)
    q_mask = q_offsets < N_CTX
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    # Initialize max and sum for online softmax
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    
    # Loop over K blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets < N_CTX
        
        # Load Q block
        Q_block_ptr = tl.make_block_ptr(
            base=Q,
            shape=(N_CTX, D_HEAD),
            strides=(stride_qm, stride_qk),
            offsets=(q_start, 0),
            block_shape=(BLOCK_M, D_HEAD),
            order=(1, 0)
        )
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Load K block
        K_block_ptr = tl.make_block_ptr(
            base=K,
            shape=(D_HEAD, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, start_n),
            block_shape=(D_HEAD, BLOCK_N),
            order=(0, 1)
        )
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Compute attention scores: QK^T / sqrt(d)
        attn_scores = tl.dot(q, k)
        attn_scores = attn_scores * (1.0 / tl.sqrt(tl.cast(D_HEAD, tl.float32)))
        
        # Apply causal mask if needed
        if STAGE == 3:  # Causal attention
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            attn_scores = tl.where(causal_mask, attn_scores, float("-inf"))
        
        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(attn_scores, axis=1))
        p = tl.exp(m_i - m_ij)
        l_i = p * l_i + tl.sum(tl.exp(attn_scores - m_ij[:, None]), axis=1)
        
        # Update m_i
        m_i = m_ij
        
        # Load V block
        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(N_CTX, D_HEAD),
            strides=(stride_vk, stride_vn),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, D_HEAD),
            order=(1, 0)
        )
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Compute attention weights
        attn_weights = tl.exp(attn_scores - m_ij[:, None])
        
        # Accumulate weighted V
        acc = acc * (p * l_i[:, None] / l_i[:, None])  # This is simplified; actual implementation is more complex
        acc = acc + tl.dot(attn_weights.to(v.dtype), v)
    
    # Normalize
    acc = acc / l_i[:, None]
    
    # Store output
    Out_block_ptr = tl.make_block_ptr(
        base=Out,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_on),
        offsets=(q_start, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    tl.store(Out_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def _flash_attention_kernel(
    Q, K, V, Out,
    L,  # Log sum of exp
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kk, stride_kn,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh, stride_lm,
    Z, H, N_CTX, D_HEAD,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # FlashAttention-style implementation
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Offsets
    off_hz = pid_h + pid_m * H
    q_start = pid_m * BLOCK_M
    q_offsets = q_start + tl.arange(0, BLOCK_M)
    q_mask = q_offsets < N_CTX
    
    # Initialize state
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    # Load Q block
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, D_HEAD),
        strides=(stride_qm, stride_qk),
        offsets=(q_start, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
    # Loop over K/V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets < N_CTX
        
        # Load K block
        K_block_ptr = tl.make_block_ptr(
            base=K,
            shape=(D_HEAD, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, start_n),
            block_shape=(D_HEAD, BLOCK_N),
            order=(0, 1)
        )
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Load V block
        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(N_CTX, D_HEAD),
            strides=(stride_vk, stride_vn),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, D_HEAD),
            order=(1, 0)
        )
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Compute attention scores
        qk = tl.dot(q, k)  # [BLOCK_M, BLOCK_N]
        qk = qk * (1.0 / tl.sqrt(tl.cast(D_HEAD, tl.float32)))
        
        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Compute max for numerical stability
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(m_i - m_ij)
        l_i_new = p * l_i + tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)
        
        # Update accumulator
        acc = acc * (p * l_i / l_i_new)[:, None]
        w = tl.exp(qk - m_ij[:, None])
        acc = acc + tl.dot(w.to(v.dtype), v)
        
        # Update state
        m_i = m_ij
        l_i = l_i_new
        
        # Store intermediate results (optional)
        # L_block_ptr = tl.make_block_ptr(
        #     base=L,
        #     shape=(N_CTX,),
        #     strides=(stride_lm,),
        #     offsets=(q_start,),
        #     block_shape=(BLOCK_M,),
        #     order=(0,)
        # )
        # tl.store(L_block_ptr, m_i + tl.log(l_i), mask=q_mask)
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    Out_block_ptr = tl.make_block_ptr(
        base=Out,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_on),
        offsets=(q_start, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0)
    )
    tl.store(Out_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))

def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Triton-based flash attention implementation"""
    
    # Ensure inputs are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Set softmax scale
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Output tensor
    Out = torch.empty_like(Q)
    
    # Grid configuration
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
    num_stages = 1 if head_dim > 64 else 2
    
    # Calculate grid size
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)
    
    # Launch kernel
    _flash_attention_kernel[grid](
        Q, K, V, Out,
        None,  # L (logsumexp) - not needed for forward pass
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return Out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Use our custom Triton flash attention
        return flash_attention(Q, K, V)