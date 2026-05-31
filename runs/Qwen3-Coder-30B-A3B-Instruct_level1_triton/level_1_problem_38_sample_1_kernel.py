import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_mean_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
        
    # Calculate the starting offset for this row
    row_start = row_idx * n_cols
    
    # Shared memory for reduction
    shared_sum = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Initialize sum
    sum_val = 0.0
    
    # Process elements in chunks
    for i in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual offset
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_cols
        
        # Load absolute values
        x_vals = tl.load(x_ptr + row_start + offset, mask=mask, other=0.0)
        abs_vals = tl.abs(x_vals)
        
        # Accumulate sum
        sum_val += tl.sum(abs_vals)
        
    # Store in shared memory
    shared_sum[tl.program_id(1)] = sum_val
    
    # Synchronize threads
    tl.sync()
    
    # Reduce within block
    if tl.program_id(1) == 0:
        # Sum all partial sums from different blocks
        final_sum = 0.0
        for i in range(tl.num_programs(1)):
            final_sum += shared_sum[i]
        
        # Store mean
        mean_val = final_sum / n_cols
        tl.store(out_ptr + row_idx, mean_val)

@triton.jit
def l1_norm_divide_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
        
    # Calculate the starting offset for this row
    row_start = row_idx * n_cols
    
    # Load mean value for this row
    mean_val = tl.load(mean_ptr + row_idx)
    
    # Process elements in chunks
    for i in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual offset
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_cols
        
        # Load input values
        x_vals = tl.load(x_ptr + row_start + offset, mask=mask, other=0.0)
        
        # Divide by mean
        result = x_vals / mean_val
        
        # Store result
        tl.store(out_ptr + row_start + offset, result, mask=mask)

def triton_l1_norm(x: torch.Tensor):
    """
    Performs L1 normalization using Triton kernels.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Calculate mean for each row
    means = torch.empty(batch_size, dtype=torch.float32, device='cuda')
    
    # Kernel configuration
    BLOCK_SIZE = 1024
    grid_rows = batch_size
    grid_cols = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # First kernel: compute mean for each row
    l1_norm_mean_kernel[(grid_rows, grid_cols),](x, means, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second kernel: divide each element by its row's mean
    l1_norm_divide_kernel[(grid_rows,),](x, means, out, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using Triton kernels.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return triton_l1_norm(x)