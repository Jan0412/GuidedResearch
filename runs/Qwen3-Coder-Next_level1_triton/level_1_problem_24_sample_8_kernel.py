import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    X_ptr,  # Pointer to input
    Y_ptr,  # Pointer to output
    batch_size,  # Number of rows
    dim,  # Dimension of each row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch dimension)
    batch_id = tl.program_id(0)
    
    # Calculate base pointers for this row
    row_start = batch_id * dim
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load the row data in chunks
    row_max = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)
    row_sum = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float32)
    
    # First pass: compute max and exponential sum with online algorithm
    for start in range(0, dim, BLOCK_SIZE):
        # Create offsets for this chunk
        chunk_offsets = start + offsets
        mask = chunk_offsets < dim
        
        # Load input data
        x = tl.load(X_ptr + row_start + chunk_offsets, mask=mask, other=-float("inf")).to(tl.float32)
        
        # Update max with online algorithm
        row_max = tl.maximum(row_max, x)
        
        # Update sum with online algorithm
        exp_x = tl.exp(x - row_max)
        row_sum = row_sum * tl.exp(row_max - tl.maximum(row_max, x)) + exp_x
        
        # Update max again after computing exp
        row_max = tl.maximum(row_max, x)
    
    # Finalize max and sum for this row
    row_max = tl.max(row_max, axis=0)
    row_sum = tl.sum(row_sum * tl.exp(row_max - row_max), axis=0)
    log_sum = tl.log(row_sum)
    
    # Second pass: compute log_softmax
    for start in range(0, dim, BLOCK_SIZE):
        chunk_offsets = start + offsets
        mask = chunk_offsets < dim
        
        # Load input data
        x = tl.load(X_ptr + row_start + chunk_offsets, mask=mask, other=-float("inf")).to(tl.float32)
        
        # Compute log_softmax: x - max - log(sum(exp(x - max)))
        log_softmax_val = x - row_max - log_sum
        
        # Store result
        tl.store(Y_ptr + row_start + chunk_offsets, log_softmax_val.to(X_ptr.dtype.element_ty), mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int = 1):
    """
    Triton implementation of log_softmax.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute log_softmax
        
    Returns:
        Tensor with log_softmax applied
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert dim == 1, "Only dim=1 is supported for this implementation"
    
    x = x.contiguous()
    output = torch.empty_like(x)
    
    batch_size = x.size(0)
    dim_size = x.size(1)
    
    # Use a block size that works well for the dimension
    BLOCK_SIZE = 512
    
    # Grid: one block per batch element
    grid = (batch_size,)
    
    # Launch the kernel
    log_softmax_kernel[grid](
        x, output, batch_size, dim_size, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs LogSoftmax activation using Triton kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return triton_log_softmax(x, dim=self.dim)