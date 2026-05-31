import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def scaled_dot_product_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seq_len, head_dim,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr
):
    # Get the batch and head indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Initialize pointers to Q, K, V
    q_ptr = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_ptr = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_ptr = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    out_ptr = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    
    # Loop over the sequence length dimension
    for m in range(0, seq_len, BLOCK_M):
        # Create m_offset for Q
        m_offset = m + tl.arange(0, BLOCK_M)
        q_mask = m_offset < seq_len
        
        # Load Q
        q = tl.load(q_ptr + m_offset[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :] * stride_qk, 
                    mask=q_mask[:, None], other=0.0)
        
        # Compute attention scores
        attn_scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        
        # Loop over the key dimension (K)
        for n in range(0, seq_len, BLOCK_N):
            # Create n_offset for K and V
            n_offset = n + tl.arange(0, BLOCK_N)
            k_mask = n_offset < seq_len
            
            # Load K
            k = tl.load(k_ptr + n_offset[None, :] * stride_km + tl.arange(0, HEAD_DIM)[None, :] * stride_kk, 
                        mask=k_mask[None, :], other=0.0)
            
            # Compute dot product
            attn_scores += tl.dot(q, k, trans_b=True) * SCALE
            
        # Apply softmax
        max_scores = tl.max(attn_scores, axis=1, keepdims=True)
        exp_scores = tl.exp(attn_scores - max_scores)
        sum_exp_scores = tl.sum(exp_scores, axis=1, keepdims=True)
        probs = exp_scores / sum_exp_scores
        
        # Compute output
        for n in range(0, seq_len, BLOCK_N):
            n_offset = n + tl.arange(0, BLOCK_N)
            v_mask = n_offset < seq_len
            
            # Load V
            v = tl.load(v_ptr + n_offset[None, :] * stride_vm + tl.arange(0, HEAD_DIM)[None, :] * stride_vk, 
                        mask=v_mask[None, :], other=0.0)
            
            # Compute output contribution
            out_contrib = tl.dot(probs, v)
            
            # Store output
            out_offset = m + tl.arange(0, BLOCK_M)
            out_mask = out_offset < seq_len
            
            tl.store(out_ptr + out_offset[:, None] * stride_om + tl.arange(0, HEAD_DIM)[None, :] * stride_ok,
                     out_contrib, mask=out_mask[:, None])

def triton_scaled_dot_product_attention(Q, K, V):
    # Ensure inputs are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Create output tensor
    out = torch.empty_like(Q)
    
    # Calculate scale factor
    scale = 1.0 / math.sqrt(head_dim)
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    # Grid configuration
    grid = (batch_size, num_heads)
    
    # Launch kernel
    scaled_dot_product_attention_kernel[grid](
        Q, K, V, out,
        seq_len, head_dim,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        HEAD_DIM=head_dim,
        SCALE=scale
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_scaled_dot_product_attention(Q, K, V)