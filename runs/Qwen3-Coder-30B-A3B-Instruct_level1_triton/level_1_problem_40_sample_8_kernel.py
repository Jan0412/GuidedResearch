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
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate the starting offset for this row
    row_start = row_idx * N
    
    # Load input data for this row
    x = tl.load(x_ptr + row_start + tl.arange(0, BLOCK_SIZE), mask=row_start + tl.arange(0, BLOCK_SIZE) < row_idx * N + N, other=0.0)
    
    # Compute mean
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / N
    
    # Store mean for this row
    tl.store(mean_ptr + row_idx, mean)
    
    # Compute variance
    x_centered = x - mean
    x_squared = x_centered * x_centered
    sum_sq = tl.sum(x_squared, axis=0)
    var = sum_sq / N
    
    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store reciprocal standard deviation
    tl.store(rstd_ptr + row_idx, rstd)
    
    # Normalize and apply scale/shift
    x_norm = x_centered * rstd
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0)
    
    out = x_norm * weight + bias
    
    # Store output
    tl.store(out_ptr + row_start + tl.arange(0, BLOCK_SIZE), out, mask=row_start + tl.arange(0, BLOCK_SIZE) < row_idx * N + N)

def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of Layer Normalization
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert weight.is_cuda, "Weight tensor must be on CUDA."
    assert bias.is_cuda, "Bias tensor must be on CUDA."
    
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    batch_size = x.shape[0]
    feature_size = x.shape[1]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Prepare mean and rstd tensors
    mean = torch.empty(batch_size, dtype=torch.float32, device=x.device)
    rstd = torch.empty(batch_size, dtype=torch.float32, device=x.device)
    
    # Calculate grid
    BLOCK_SIZE = 128
    grid = lambda meta: (batch_size,)
    
    # Launch kernel
    layer_norm_kernel[grid](
        x,
        weight,
        bias,
        out,
        mean,
        rstd,
        feature_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return triton_layer_norm(x, self.weight, self.bias)