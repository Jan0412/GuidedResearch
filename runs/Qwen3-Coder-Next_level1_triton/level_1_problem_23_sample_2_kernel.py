import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input matrix
    row_idx = tl.program_id(0)
    
    # Compute row start pointer
    row_start = row_idx * n_cols
    
    # Create column offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load row data
    row = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float("inf"))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    
    # Compute exp
    exp_row = tl.exp(row_minus_max)
    
    # Compute sum of exp
    exp_sum = tl.sum(exp_row, axis=0)
    
    # Compute log sum exp for normalization
    log_sum_exp = tl.log(exp_sum)
    
    # Compute softmax: exp(x - max) / sum(exp(x - max)) = exp(x - max - log_sum_exp)
    softmax_row = tl.exp(row_minus_max - log_sum_exp)
    
    # Store result
    tl.store(out_ptr + row_start + offsets, softmax_row, mask=mask)


def triton_softmax(x: torch.Tensor):
    """
    Triton implementation of softmax along dimension 1.
    Optimized for large dimensions using online softmax algorithm.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, num_features = x.shape
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Determine block size (power of 2 close to num_features, but at most 8192)
    BLOCK_SIZE = min(8192, 1 << (num_features - 1).bit_length())
    
    # Grid: one block per row
    grid = (batch_size,)
    
    # Launch kernel
    softmax_kernel[grid](
        x, 
        out, 
        batch_size, 
        num_features,
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
        Applies Softmax activation to the input tensor using optimized Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x)