import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    eps: tl.constexpr
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate the start position for this row
    row_start = row_idx * N
    
    # Each program processes one row
    # Compute mean
    mean = 0.0
    for i in range(0, N, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < N
        x_vals = tl.load(x_ptr + row_start + off, mask=mask, other=0.0)
        mean += tl.sum(x_vals)
    
    mean = mean / N
    
    # Compute variance
    var = 0.0
    for i in range(0, N, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < N
        x_vals = tl.load(x_ptr + row_start + off, mask=mask, other=0.0)
        diff = x_vals - mean
        var += tl.sum(diff * diff)
    
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and rstd for this row
    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)
    
    # Normalize and apply scale/shift
    for i in range(0, N, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < N
        x_vals = tl.load(x_ptr + row_start + off, mask=mask, other=0.0)
        normalized = (x_vals - mean) * rstd
        weight_vals = tl.load(weight_ptr + off, mask=mask, other=0.0)
        bias_vals = tl.load(bias_ptr + off, mask=mask, other=0.0)
        output_vals = normalized * weight_vals + bias_vals
        tl.store(output_ptr + row_start + off, output_vals, mask=mask)

class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernels for Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with optimized Triton implementation.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Flatten the input to process each row independently
        original_shape = x.shape
        batch_dims = original_shape[:-len(self.normalized_shape)]
        num_rows = 1
        for d in batch_dims:
            num_rows *= d
        N = 1
        for d in self.normalized_shape:
            N *= d
            
        x_flat = x.view(num_rows, N)
        
        # Allocate output tensor
        output = torch.empty_like(x_flat)
        
        # Allocate intermediate tensors for mean and rstd
        means = torch.empty(num_rows, dtype=torch.float32, device=x.device)
        rstds = torch.empty(num_rows, dtype=torch.float32, device=x.device)
        
        # Configure kernel launch parameters
        BLOCK_SIZE = 1024
        GROUP_SIZE_M = 8
        
        # Calculate grid size
        grid = lambda meta: (num_rows,)
        
        # Launch the Triton kernel
        layer_norm_kernel[grid](
            x_flat,
            self.weight,
            self.bias,
            output,
            means,
            rstds,
            N,
            num_rows,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M,
            eps=self.eps
        )
        
        # Reshape back to original shape
        return output.view(original_shape)