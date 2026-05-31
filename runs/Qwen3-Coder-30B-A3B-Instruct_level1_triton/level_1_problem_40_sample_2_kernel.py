import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    H,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate the starting position for this row
    x_row_start = row_idx * N
    output_row_start = row_idx * N
    
    # Load data for this row
    x = tl.load(x_ptr + x_row_start + tl.arange(0, BLOCK_SIZE), mask=(tl.arange(0, BLOCK_SIZE) < N))
    
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    
    # Store mean for this row
    tl.store(mean_ptr + row_idx, mean)
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    
    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + EPSILON)
    
    # Store reciprocal standard deviation for this row
    tl.store(rstd_ptr + row_idx, rstd)
    
    # Normalize and scale
    x_normalized = x_centered * rstd
    
    # Load weight and bias
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=(tl.arange(0, BLOCK_SIZE) < N))
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=(tl.arange(0, BLOCK_SIZE) < N))
    
    # Apply normalization, scaling, and bias
    output = x_normalized * weight + bias
    
    # Store result
    tl.store(output_ptr + output_row_start + tl.arange(0, BLOCK_SIZE), output, mask=(tl.arange(0, BLOCK_SIZE) < N))

def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of LayerNorm
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    assert x.dtype == torch.float32, "Only FP32 supported"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    batch_size = x.shape[0]
    hidden_size = x.shape[1]
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Allocate intermediate tensors for mean and rstd
    mean = torch.empty(batch_size, dtype=torch.float32, device=x.device)
    rstd = torch.empty(batch_size, dtype=torch.float32, device=x.device)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = lambda meta: (batch_size,)
    
    # Launch kernel
    layer_norm_kernel[grid](
        x,
        weight,
        bias,
        output,
        mean,
        rstd,
        hidden_size,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
        EPSILON=eps
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernels for Layer Normalization.
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