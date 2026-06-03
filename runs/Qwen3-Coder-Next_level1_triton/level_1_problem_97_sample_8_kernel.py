import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scaled_dot_product_attention_kernel(
    Q_ptr,  # [B, H, L, D]
    K_ptr,  # [B, H, L, D]
    V_ptr,  # [B, H, L, D]
    out_ptr,  # [B, H, L, D]
    # Batch, heads, sequence, and embedding dimensions
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    # Scaling factor (1/sqrt(D))
    inv_sqrt_d: tl.constexpr,
    # Block sizes for tiling
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Program IDs: [batch_idx, head_idx, seq_block_idx, q_block_idx]
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_block_idx = tl.program_id(2)  # For output sequence position block
    
    # Compute base pointers for this [batch, head] pair
    base_offset = (batch_idx * H + head_idx) * (L * D)
    Q_ptr += base_offset
    K_ptr += base_offset
    V_ptr += base_offset
    out_ptr += base_offset
    
    # Compute Q block start index
    q_start = seq_block_idx * BLOCK_L
    q_offsets = q_start + tl.arange(0, BLOCK_L)
    q_mask = q_offsets < L
    
    # Load Q block: [BLOCK_L, D]
    q_block = tl.load(
        Q_ptr + q_offsets[:, None] * D + tl.arange(0, BLOCK_D)[None, :],
        mask=q_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < D),
        other=0.0
    )
    
    # Initialize accumulator for O = softmax(QK^T) * V
    # We need to compute for each query position: 
    #   out[i] = sum_j( exp(q[i]*k[j]^T / sqrt(d)) * v[j] ) / sum_j( exp(q[i]*k[j]^T / sqrt(d)) )
    # We'll use online softmax to compute this stably
    
    # Initialize max and sum for softmax
    row_max = tl.full([BLOCK_L], -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_L], dtype=tl.float32)
    acc = tl.zeros([BLOCK_L, BLOCK_D], dtype=tl.float32)
    
    # Iterate over key/value sequence positions in blocks
    for k_start in range(0, L, BLOCK_L):
        k_offsets = k_start + tl.arange(0, BLOCK_L)
        kv_mask = k_offsets < L
        
        # Load K block: [BLOCK_L, D]
        k_block = tl.load(
            K_ptr + k_offsets[:, None] * D + tl.arange(0, BLOCK_D)[None, :],
            mask=kv_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < D),
            other=0.0
        )
        
        # Compute QK^T block: [BLOCK_L, BLOCK_L] = [BLOCK_L, D] @ [D, BLOCK_L]
        # But we only need diagonal blocks for causal attention, and full for non-causal
        # Here we implement non-causal attention (standard attention)
        qk = tl.dot(q_block.to(tl.float16), k_block.T.to(tl.float16), allow_tf32=False)
        qk = qk.to(tl.float32) * inv_sqrt_d
        
        # Compute softmax with numerical stability (online softmax)
        new_max = tl.maximum(row_max, tl.max(qk, axis=1))
        exp_diff = tl.exp(row_max - new_max)
        row_sum = row_sum * exp_diff + tl.exp(qk - new_max[:, None])
        row_max = new_max
        
        # Load V block: [BLOCK_L, D]
        v_block = tl.load(
            V_ptr + k_offsets[:, None] * D + tl.arange(0, BLOCK_D)[None, :],
            mask=kv_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < D),
            other=0.0
        )
        
        # Update accumulator: acc = acc * exp_diff + softmax(qk) @ v_block
        p = tl.exp(qk - new_max[:, None])
        acc = acc * exp_diff[:, None] + tl.dot(p.to(tl.float16), v_block.to(tl.float16), allow_tf32=False)
    
    # Final normalization: acc / row_sum
    out_block = acc / row_sum[:, None]
    
    # Store output: [BLOCK_L, D]
    tl.store(
        out_ptr + q_offsets[:, None] * D + tl.arange(0, BLOCK_D)[None, :],
        out_block.to(tl.float16),
        mask=q_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < D)
    )


def triton_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of scaled dot-product attention (non-causal).
    Supports FP16 input tensors with FP32 computation for stability.
    """
    # Ensure inputs are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Extract dimensions
    B, H, L, D = Q.shape
    
    # Compute scaling factor
    inv_sqrt_d = 1.0 / (D ** 0.5)
    
    # Create output tensor
    out = torch.empty_like(Q)
    
    # Define block sizes (tunable for performance)
    BLOCK_L = 64
    BLOCK_D = 64
    
    # Grid: [batch, heads, seq_blocks]
    grid = (B, H, (L + BLOCK_L - 1) // BLOCK_L)
    
    # Launch kernel
    scaled_dot_product_attention_kernel[grid](
        Q, K, V, out,
        B=B, H=H, L=L, D=D,
        inv_sqrt_d=inv_sqrt_d,
        BLOCK_L=BLOCK_L,
        BLOCK_D=BLOCK_D,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Replace PyTorch's scaled_dot_product_attention with our Triton implementation
        return triton_scaled_dot_product_attention(Q, K, V)