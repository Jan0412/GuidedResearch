import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr, 
    out_ptr, 
    dim, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row of the input tensor
    row_id = tl.program_id(0)
    
    # Pointers to the start of the current row in input and output tensors
    row_start_ptr = x_ptr + row_id * dim
    out_start_ptr = out_ptr + row_id * dim

    # Online softmax algorithm to compute max and sum of exponentials in one pass
    # This reduces memory reads from 3 passes to 2 passes
    m_i = -float('inf')
    s_i = 0.0

    # Pass 1: Compute the global maximum and the sum of exponentials
    for i in range(0, dim, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim
        # Load block of data; use -inf for masked values to avoid affecting the max
        x_block = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        
        # Local max of the current block
        m_block = tl.max(x_block, axis=0)
        
        # Update global max
        m_next = tl.maximum(m_i, m_block)
        
        # Update global sum of exponentials using the online softmax formula:
        # s_next = s_i * exp(m_i - m_next) + sum(exp(x_block - m_next))
        s_i = s_i * tl.exp(m_i - m_next) + tl.sum(tl.exp(x_block - m_next), axis=0)
        m_i = m_next

    # Pass 2: Compute the final softmax values and store them
    for i in range(0, dim, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim
        x_block = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        
        # Softmax formula: exp(x - max) / sum(exp(x - max))
        out_block = tl.exp(x_block - m_i) / s_i
        tl.store(out_start_ptr + cols, out_block, mask=mask)


def triton_softmax(x: torch.Tensor):
    """
    Wrapper function to launch the Triton softmax kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure tensor is contiguous for pointer arithmetic
    x = x.contiguous()
    
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    
    # BLOCK_SIZE is a power of 2. 1024 is generally a good default for FP32.
    BLOCK_SIZE = 1024
    
    # Grid is (batch_size,) because we process one row per program
    grid = (batch_size,)
    
    softmax_kernel[grid](
        x, 
        out, 
        dim, 
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