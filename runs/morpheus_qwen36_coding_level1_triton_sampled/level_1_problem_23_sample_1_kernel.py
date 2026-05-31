import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    stride_x_row,
    stride_out_row,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Softmax using online algorithm.
    Processes one row per program.
    """
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the row
    x_row_ptr = x_ptr + row_idx * stride_x_row
    out_row_ptr = out_ptr + row_idx * stride_out_row
    
    # Online Softmax state
    m = -float('inf')
    s = 0.0
    
    # Pass 1: Compute running max and sum using online algorithm
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        # Load block of data
        x_block = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
        
        # Update max
        m_block = tl.max(x_block, axis=0)
        m_new = tl.maximum(m, m_block)
        
        # Update sum
        # s = s * exp(m - m_new) + sum(exp(x_block - m_new))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x_block - m_new), axis=0)
        
        m = m_new
    
    # Pass 2: Compute output and store
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        # Load block again (or we could store exp values, but reloading is memory efficient enough)
        x_block = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
        
        # Compute softmax: exp(x - m) / s
        out_block = tl.exp(x_block - m) / s
        
        # Store result
        tl.store(out_row_ptr + offsets, out_block, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton softmax kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    
    # Tunable block size
    BLOCK_SIZE = 1024
    
    # Grid configuration: one program per row
    grid = (batch_size,)
    
    # Launch kernel
    softmax_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        stride_x_row=x.stride(0),
        stride_out_row=out.stride(0),
        batch_size=batch_size,
        dim=dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x)