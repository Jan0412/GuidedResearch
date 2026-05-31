import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    x_ptr, 
    out_ptr, 
    dim, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the batch
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the current row
    x_row_ptr = x_ptr + row_idx * dim
    out_row_ptr = out_ptr + row_idx * dim

    # Pass 1: Online Max and Sum of Exponentials
    # We use the online softmax algorithm to compute max and sum in one pass
    m = -float('inf')
    s = 0.0
    
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        # Load a block of the row
        x_block = tl.load(x_row_ptr + offsets, mask=mask, other=-float('inf'))
        
        # Local max of the current block
        m_block = tl.max(x_block, axis=0)
        
        # Update global max for the row
        m_new = tl.maximum(m, m_block)
        
        # Update global sum of exponentials
        # s_new = s_old * exp(m_old - m_new) + sum(exp(x_block - m_new))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x_block - m_new), axis=0)
        m = m_new
    
    # Log-Sum-Exp = m + log(s)
    log_sum_exp = m + tl.log(s)

    # Pass 2: Compute LogSoftmax and store result
    # LogSoftmax(x) = x - LogSumExp
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x_block = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_row_ptr + offsets, x_block - log_sum_exp, mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int):
    """
    Triton wrapper for LogSoftmax operation.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    # The kernel is designed for reduction over the last dimension (dim=1 in this case)
    # If dim is not the last dimension, this would need adjustment.
    batch_size, d_size = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    # Grid is one program per row
    grid = (batch_size,)
    
    log_softmax_kernel[grid](
        x, out, d_size, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a LogSoftmax activation using a custom Triton kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        # Assuming input is (batch_size, dim) and reduction is on dim=1
        return triton_log_softmax(x, self.dim)