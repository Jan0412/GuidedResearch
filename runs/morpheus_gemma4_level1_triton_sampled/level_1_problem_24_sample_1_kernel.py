import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    x_ptr, 
    out_ptr, 
    dim_size, 
    stride_row, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_id = tl.program_id(0)
    row_offset = row_id * stride_row
    
    # Initialize variables for online LogSumExp calculation
    # m_n = max(m_{n-1}, x_n)
    # s_n = s_{n-1} * exp(m_{n-1} - m_n) + sum(exp(x_i - m_n))
    running_max = -float('inf')
    running_sum = 0.0
    
    # First pass: compute the LogSumExp (LSE) across the row
    # Since dim_size can be very large, we iterate in blocks
    num_blocks = (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, num_blocks):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load a chunk of the row
        chunk = tl.load(x_ptr + row_offset + offsets, mask=mask, other=-float('inf'))
        
        # Local reduction for the current block
        chunk_max = tl.max(chunk, axis=0)
        # Use a safe max to avoid NaN during subtraction if the chunk is all -inf
        safe_max = tl.where(chunk_max == -float('inf'), 0.0, chunk_max)
        chunk_sum = tl.sum(tl.exp(chunk - safe_max), axis=0)
        
        # Update running max and running sum using the online softmax formula
        new_max = tl.maximum(running_max, chunk_max)
        
        # Correct for the change in max: s_new = s_old * exp(m_old - m_new) + s_chunk * exp(m_chunk - m_new)
        # Note: if running_max is -inf, the first term becomes 0.0 * 0.0 = 0.0
        running_sum = running_sum * tl.exp(running_max - new_max) + \
                      chunk_sum * tl.exp(safe_max - new_max)
        running_max = new_max
        
    # Compute final LSE: LSE = max + log(sum_exp)
    lse = running_max + tl.log(running_sum)
    
    # Second pass: compute output = x - LSE
    for i in range(0, num_blocks):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        chunk = tl.load(x_ptr + row_offset + offsets, mask=mask)
        tl.store(out_ptr + row_offset + offsets, chunk - lse, mask=mask)

def triton_log_softmax(x: torch.Tensor):
    """
    Triton wrapper for LogSoftmax activation.
    Assumes reduction is over the last dimension (dim=1 for 2D input).
    """
    batch_size, dim_size = x.shape
    # Ensure tensor is contiguous for pointer arithmetic
    x = x.contiguous()
    out = torch.empty_like(x)
    
    stride_row = x.stride(0)
    # BLOCK_SIZE is tuned for typical GPU shared memory/register limits
    BLOCK_SIZE = 1024
    
    # Grid: one program per row
    grid = (batch_size,)
    
    log_softmax_kernel[grid](
        x, out, dim_size, stride_row, 
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
        # The kernel is optimized for reduction over the second dimension (axis=1)
        return triton_log_softmax(x)