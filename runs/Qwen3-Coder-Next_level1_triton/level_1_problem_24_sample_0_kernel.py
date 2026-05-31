import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate the starting offset for this row
    row_start = row_idx * n_cols
    
    # Load the row data
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load the row
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-tl.inf)
    
    # Compute max for numerical stability
    row_max = tl.max(x, axis=0)
    
    # Subtract max for numerical stability
    x_shifted = x - row_max
    
    # Compute exponentials
    exp_x = tl.exp(x_shifted)
    
    # Compute sum of exponentials
    exp_sum = tl.sum(exp_x, axis=0)
    
    # Compute log sum using log-sum-exp trick
    log_sum = tl.log(exp_sum) + row_max
    
    # Compute log softmax: x - log_sum
    result = x - log_sum
    
    # Store the result
    tl.store(output_ptr + row_start + offsets, result, mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int = -1):
    """
    Triton implementation of log softmax.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply log softmax
        
    Returns:
        Tensor with log softmax applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Make tensor contiguous
    x = x.contiguous()
    
    # Get shape info
    shape = x.shape
    dim = dim if dim >= 0 else len(shape) + dim
    
    # Reshape to 2D if needed: (batch_size, features)
    if dim != len(shape) - 1:
        # Permute dimensions to move the target dimension to the end
        perm = list(range(len(shape)))
        perm.pop(dim)
        perm.append(dim)
        x = x.permute(perm)
        shape_perm = x.shape
    
    # Reshape to 2D
    batch_size = 1
    for i in range(len(shape) - 1):
        batch_size *= shape[i]
    features = shape[-1]
    
    x_2d = x.view(batch_size, features)
    
    # Prepare output tensor
    output = torch.empty_like(x_2d)
    
    # Set block size - choose optimal size based on feature dimension
    BLOCK_SIZE = min(1024, triton.next_power_of_2(features))
    
    # Grid: one block per row
    grid = (batch_size,)
    
    # Launch the kernel
    log_softmax_kernel[grid](
        x_2d,
        output,
        batch_size,
        features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original shape if needed
    if dim != len(shape) - 1:
        # Reverse the permutation
        inv_perm = [0] * len(shape)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        output = output.view(shape_perm).permute(inv_perm)
    
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
        Applies LogSoftmax activation to the input tensor using optimized Triton kernel.
        """
        return triton_log_softmax(x, dim=self.dim)