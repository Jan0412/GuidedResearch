import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # stride to the next row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_start = tl.program_id(0) * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    # Load input data
    x = tl.load(X + row_start + col_offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x * mask, axis=0) / tl.sum(mask.to(tl.float32), axis=0)
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered * mask, axis=0) / tl.sum(mask.to(tl.float32), axis=0)
    
    # Compute standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and rstd for backward pass (if needed)
    tl.store(Mean + tl.program_id(0), mean)
    tl.store(Rstd + tl.program_id(0), rstd)
    
    # Normalize and apply scale/bias
    # Load weights and biases
    w = tl.load(W + col_offsets, mask=mask, other=0.0)
    b = tl.load(B + col_offsets, mask=mask, other=0.0)
    
    # Normalize
    x_hat = x_centered * rstd
    
    # Apply scale and shift
    out = x_hat * w + b
    
    # Store output
    tl.store(Y + row_start + col_offsets, out, mask=mask)


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Applies Layer Normalization using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, *normalized_shape)
        weight: Scale parameter of shape (normalized_shape)
        bias: Shift parameter of shape (normalized_shape)
        eps: Small value to avoid division by zero
    
    Returns:
        Output tensor of same shape as input
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get normalized shape dimensions
    *batch_dims, normalized_shape = x.shape
    normalized_size = weight.numel()
    
    # Reshape to 2D: (batch_size * prod(batch_dims), normalized_shape)
    x_2d = x.view(-1, normalized_size)
    batch_size_flat = x_2d.size(0)
    
    # Prepare output tensor
    y = torch.empty_like(x)
    y_2d = y.view(-1, normalized_size)
    
    # Prepare mean and rstd tensors (for potential backward pass)
    mean = torch.empty(batch_size_flat, dtype=torch.float32, device=x.device)
    rstd = torch.empty(batch_size_flat, dtype=torch.float32, device=x.device)
    
    # Calculate grid size
    BLOCK_SIZE = 1024  # Tunable parameter
    
    # Launch kernel
    grid = (batch_size_flat,)
    layer_norm_kernel[grid](
        x_2d, y_2d, weight, bias, mean, rstd,
        normalized_size, normalized_size, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output back to original shape
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using Triton kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        # Create parameters for LayerNorm
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