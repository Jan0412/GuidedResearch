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
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    x_ptr += row_idx * stride_row
    out_ptr += row_idx * stride_row

    # Online softmax implementation to reduce memory passes
    # Pass 1: Compute global maximum and sum of exponentials
    m_i = -float('inf')
    d_i = 0.0
    
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        # Load chunk of the row
        x_chunk = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        
        # Local max and sum for the current chunk
        m_chunk = tl.max(x_chunk, axis=0)
        d_chunk = tl.sum(tl.exp(x_chunk - m_chunk), axis=0)
        
        # Update global max and sum using online softmax formula
        m_next = tl.maximum(m_i, m_chunk)
        d_i = d_i * tl.exp(m_i - m_next) + d_chunk * tl.exp(m_chunk - m_next)
        m_i = m_next

    # Pass 2: Compute the final softmax values and store them
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_chunk = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        
        # Normalize using the global max and sum computed in Pass 1
        out_chunk = tl.exp(x_chunk - m_i) / d_i
        tl.store(out_ptr + offsets, out_chunk, mask=mask)

def triton_softmax(x: torch.Tensor):
    """
    Wrapper for the Triton softmax kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure the tensor is contiguous for pointer arithmetic
    x = x.contiguous()
    batch_size, n_cols = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Calculate the stride between rows
    stride_row = x.stride(0)
    
    # Block size must be a power of 2
    BLOCK_SIZE = 1024
    
    # Grid: one program per row
    grid = (batch_size,)
    
    # Launch kernel
    softmax_kernel[grid](
        x, out, stride_row, n_cols, 
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