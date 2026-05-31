import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr, 
    out_ptr, 
    stride_row, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the current row
    row_start_ptr = x_ptr + row_idx * stride_row
    out_start_ptr = out_ptr + row_idx * stride_row

    # Online Softmax: First pass to find max and sum of exponentials
    # Initialize max and sum
    m_i = -1e38  # Use a very small number to avoid NaN in first iteration
    d_i = 0.0

    for k in range(0, n_cols, BLOCK_SIZE):
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        # Load current block
        x = tl.load(row_start_ptr + offsets, mask=mask, other=-1e38)
        
        # Local max of the block
        m_curr = tl.max(x, axis=0)
        
        # Update global max and sum using online softmax formula
        m_next = tl.maximum(m_i, m_curr)
        d_i = d_i * tl.exp(m_i - m_next) + tl.sum(tl.exp(x - m_next), axis=0)
        m_i = m_next

    # Second pass to compute and store the final softmax values
    for k in range(0, n_cols, BLOCK_SIZE):
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(row_start_ptr + offsets, mask=mask, other=-1e38)
        
        # Normalize: exp(x - max) / sum
        out = tl.exp(x - m_i) / d_i
        tl.store(out_start_ptr + offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor):
    """
    Triton wrapper for the softmax kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    batch_size, n_cols = x.shape
    out = torch.empty_like(x)
    
    # stride_row is the distance between the start of two consecutive rows
    stride_row = x.stride(0)
    
    # Block size for processing columns within a row
    BLOCK_SIZE = 1024
    
    # Grid: one program per row
    grid = (batch_size,)
    
    softmax_kernel[grid](
        x, 
        out, 
        stride_row, 
        n_cols, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Softmax activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x)