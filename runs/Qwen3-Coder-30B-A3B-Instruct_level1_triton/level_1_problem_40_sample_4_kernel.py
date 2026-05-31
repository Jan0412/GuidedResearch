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
    out_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    H,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPS: tl.constexpr
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate start position for this thread block
    start_pos = row_idx * N
    
    # Initialize accumulators for mean and variance calculation
    mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load data for this row
    x = tl.load(x_ptr + start_pos + tl.arange(0, BLOCK_SIZE), mask=(start_pos + tl.arange(0, BLOCK_SIZE)) < (row_idx + 1) * N)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    
    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    
    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + EPS)
    
    # Store mean and rstd for later use
    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)
    
    # Normalize and apply scale and bias
    x_norm = (x - mean) * rstd
    
    # Apply weight and bias
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N)
    
    out = x_norm * weight + bias
    
    # Store result
    tl.store(out_ptr + start_pos + tl.arange(0, BLOCK_SIZE), out, mask=(start_pos + tl.arange(0, BLOCK_SIZE)) < (row_idx + 1) * N)

@triton.jit
def fused_layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    H,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPS: tl.constexpr
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate start position for this thread block
    start_pos = row_idx * N
    
    # Load data for this row
    x = tl.load(x_ptr + start_pos + tl.arange(0, BLOCK_SIZE), mask=(start_pos + tl.arange(0, BLOCK_SIZE)) < (row_idx + 1) * N)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    
    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    
    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + EPS)
    
    # Store mean and rstd for later use
    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)
    
    # Normalize and apply scale and bias
    x_norm = (x - mean) * rstd
    
    # Apply weight and bias
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N)
    
    out = x_norm * weight + bias
    
    # Store result
    tl.store(out_ptr + start_pos + tl.arange(0, BLOCK_SIZE), out, mask=(start_pos + tl.arange(0, BLOCK_SIZE)) < (row_idx + 1) * N)

def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Custom Triton implementation of LayerNorm for better performance.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert weight.is_cuda, "Weight tensor must be on CUDA."
    assert bias.is_cuda, "Bias tensor must be on CUDA."
    
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Calculate dimensions
    batch_size = x.shape[0]
    num_features = x.shape[-1]
    
    # For simplicity, assume we're normalizing over the last dimension only
    # This is the typical case for LayerNorm
    
    # Determine the number of blocks needed
    BLOCK_SIZE = 1024  # Should be a power of 2 for efficiency
    GRID_SIZE = batch_size
    
    # Launch the kernel
    layer_norm_kernel[GRID_SIZE,](
        x,
        weight,
        bias,
        out,
        torch.empty(batch_size, device=x.device, dtype=torch.float32),
        torch.empty(batch_size, device=x.device, dtype=torch.float32),
        num_features,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE_M=8,
        EPS=eps
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized version of Layer Normalization using custom Triton kernels.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

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
        Applies Layer Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return triton_layer_norm(x, self.weight, self.bias, self.eps)