import torch
import torch.nn as nn
import triton
import triton.language as tl

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
    W,
    C,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr
):
    # Compute row index
    row_idx = tl.program_id(0)
    
    # Calculate offsets for current row
    x_row = x_ptr + row_idx * C
    out_row = out_ptr + row_idx * C
    
    # Load input data for this row
    x_vals = tl.load(x_row + tl.arange(0, BLOCK_SIZE), mask=(tl.arange(0, BLOCK_SIZE) < C))
    
    # Compute mean
    sum_x = tl.sum(x_vals, axis=0)
    mean = sum_x / C
    
    # Store mean for this row
    tl.store(mean_ptr + row_idx, mean)
    
    # Compute variance
    x_centered = x_vals - mean
    squared = x_centered * x_centered
    sum_squared = tl.sum(squared, axis=0)
    var = sum_squared / C
    rstd = 1.0 / tl.sqrt(var + EPS)
    
    # Store reciprocal standard deviation
    tl.store(rstd_ptr + row_idx, rstd)
    
    # Normalize and apply scale/shift
    normalized = x_centered * rstd
    weight_vals = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=(tl.arange(0, BLOCK_SIZE) < C))
    bias_vals = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=(tl.arange(0, BLOCK_SIZE) < C))
    
    out_vals = normalized * weight_vals + bias_vals
    
    # Store output
    tl.store(out_row + tl.arange(0, BLOCK_SIZE), out_vals, mask=(tl.arange(0, BLOCK_SIZE) < C))

def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of LayerNorm for 4D tensors (batch, features, dim1, dim2).
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert weight.is_cuda and bias.is_cuda, "Weight and bias tensors must be on CUDA."
    
    batch_size, features, dim1, dim2 = x.shape
    C = features * dim1 * dim2  # total channels
    
    # Flatten the input to 2D for processing
    x_flat = x.view(-1, C)
    
    # Ensure tensors are contiguous
    x_flat = x_flat.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x_flat)
    
    # Allocate memory for intermediate results (mean and rstd)
    mean = torch.empty(batch_size, dtype=torch.float32, device=x.device)
    rstd = torch.empty(batch_size, dtype=torch.float32, device=x.device)
    
    # Grid configuration
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    # Launch kernel
    layer_norm_kernel[grid](
        x_flat,
        weight,
        bias,
        out,
        mean,
        rstd,
        N=batch_size,
        H=features,
        W=dim1,
        C=C,
        BLOCK_SIZE=BLOCK_SIZE,
        EPS=eps
    )
    
    # Reshape back to original shape
    return out.view(batch_size, features, dim1, dim2)

class ModelNew(nn.Module):
    """
    Optimized Model using custom Triton kernels for Layer Normalization.
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