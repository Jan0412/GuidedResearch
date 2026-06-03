import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    batch_size, num_heads, seq_len, embed_dim,
    q_stride_b, q_stride_h, q_stride_s, q_stride_e,
    k_stride_b, k_stride_h, k_stride_s, k_stride_e,
    v_stride_b, v_stride_h, k_stride_s, v_stride_e,
    out_stride_b, out_stride_h, out_stride_s, out_stride_e,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ids
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    seq_q_id = tl.program_id(2)
    
    # Calculate base pointers for this batch, head, and query position
    q_base_ptr = Q_ptr + batch_id * q_stride_b + head_id * q_stride_h + seq_q_id * q_stride_s
    out_base_ptr = Out_ptr + batch_id * out_stride_b + head_id * out_stride_h + seq_q_id * out_stride_s
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    max_val = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process key-value pairs in blocks
    for start_k in range(0, seq_len, BLOCK_SIZE):
        # Load Q vector
        q_offsets = tl.arange(0, BLOCK_SIZE)
        q_mask = q_offsets < BLOCK_SIZE
        q_vec = tl.load(q_base_ptr + q_offsets * q_stride_e, mask=q_mask, other=0.0)
        q_vec = q_vec.to(tl.float32)
        
        # Initialize accumulator for this block
        block_max = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
        block_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Process key vectors
        for start_v in range(0, seq_len, BLOCK_SIZE):
            # Compute attention scores
            k_offsets = tl.arange(0, BLOCK_SIZE)
            k_mask = k_offsets < BLOCK_SIZE
            k_vec = tl.load(K_ptr + batch_id * k_stride_b + head_id * k_stride_h + (start_k + k_offsets) * k_stride_s, 
                           mask=k_mask, other=0.0)
            k_vec = k_vec.to(tl.float32)
            
            # Compute attention score for this Q and K
            attn_score = 0.0
            for e in range(embed_dim):
                if e < BLOCK_SIZE:
                    attn_score += q_vec[e] * k_vec[e] * scale
            
            # Update max and sum for stable softmax
            block_max = tl.maximum(block_max, attn_score)
            block_sum += tl.exp(attn_score - block_max)
        
        # Compute softmax weights
        softmax_weights = tl.exp(attn_score - block_max) / block_sum
        
        # Accumulate weighted V values
        for e in range(embed_dim):
            if e < BLOCK_SIZE:
                v_val = tl.load(V_ptr + batch_id * v_stride_b + head_id * v_stride_h + (start_k + e) * v_stride_s,
                               mask=k_mask, other=0.0)
                v_val = v_val.to(tl.float32)
                acc += softmax_weights * v_val
    
    # Store result
    out_offsets = tl.arange(0, BLOCK_SIZE)
    out_mask = out_offsets < BLOCK_SIZE
    tl.store(out_base_ptr + out_offsets * out_stride_e, acc.to(tl.float16), mask=out_mask)


class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        # Ensure contiguous tensors
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        
        # Get dimensions
        batch_size, num_heads, seq_len, embed_dim = Q.shape
        
        # Calculate strides
        q_strides = Q.stride()
        k_strides = K.stride()
        v_strides = V.stride()
        
        # Output tensor
        Out = torch.empty_like(Q)
        out_strides = Out.stride()
        
        # Scale factor for attention
        scale = 1.0 / math.sqrt(embed_dim)
        
        # Grid configuration
        grid = (batch_size, num_heads, seq_len)
        
        # Launch kernel
        attention_kernel[grid](
            Q, K, V, Out,
            batch_size, num_heads, seq_len, embed_dim,
            q_strides[0], q_strides[1], q_strides[2], q_strides[3],
            k_strides[0], k_strides[1], k_strides[2], k_strides[3],
            v_strides[0], v_strides[1], v_strides[2], v_strides[3],
            out_strides[0], out_strides[1], out_strides[2], out_strides[3],
            scale,
            BLOCK_SIZE=128,
        )
        
        return Out
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, fall back to PyTorch implementation for backward pass
        # In production, you'd want to implement custom backward kernel
        return None, None, None


def triton_attention(Q, K, V):
    return TritonAttention.apply(Q, K, V)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_attention(Q, K, V)