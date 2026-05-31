import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    X,  # Pointer to input tensor
    Y,  # Pointer to output tensor
    W,  # Pointer to weight tensor (gamma)
    B,  # Pointer to bias tensor (beta)
    Mean,  # Pointer to mean tensor (for backward, but we'll compute here for inference)
    Rstd,  # Pointer to reciprocal std tensor
    M,  # Number of rows (all dimensions except last)
    N,  # Number of columns (last dimension)
    eps,  # Epsilon for numerical stability
    stride_m,  # Stride between rows in X/Y
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input
    row_start = tl.program_id(0)
    # Compute row offset
    row = row_start * stride_m

    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load input data for this row
    x_ptrs = X + row + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / N

    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N

    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)

    # Store mean and rstd for potential use in backward pass (not used in inference)
    # For inference we can skip storing these but keeping for consistency
    tl.store(Mean + row_start, mean)
    tl.store(Rstd + row_start, rstd)

    # Normalize and scale with weight and bias
    # Load weight and bias (broadcasted across rows)
    w_ptrs = W + col_offsets
    w = tl.load(w_ptrs, mask=mask, other=0.0)
    b_ptrs = B + col_offsets
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    # Apply normalization: (x - mean) / sqrt(var + eps) * weight + bias
    out = x_centered * rstd * w + b

    # Store output
    y_ptrs = Y + row + col_offsets
    tl.store(y_ptrs, out, mask=mask)


def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Applies Layer Normalization using Triton kernel.
    
    Args:
        x: Input tensor of shape (*, normalized_shape)
        weight: Weight tensor (gamma) of shape (normalized_shape,)
        bias: Bias tensor (beta) of shape (normalized_shape,)
        eps: Epsilon for numerical stability
    
    Returns:
        Output tensor with same shape as input
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get original shape and reshape to 2D (M, N) where N is the normalized dimension
    shape = x.shape
    N = shape[-1]
    M = x.numel() // N
    x_2d = x.view(M, N)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    
    # Create mean and rstd tensors (not strictly necessary for inference but kept for consistency)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    
    # Calculate grid dimensions
    grid = (M,)
    
    # Calculate stride
    stride_m = x_2d.stride(0)
    
    # Launch the kernel
    layernorm_kernel[grid](
        x_2d, out, weight, bias, mean, rstd, M, N, eps, stride_m, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using Triton kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with Triton optimization.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        # Create parameters similar to nn.LayerNorm
        # Note: In the original Model, these would be created by nn.LayerNorm
        # Here we manually create them to match the behavior
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5  # Default eps value in nn.LayerNorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization using Triton kernel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Check if input shape matches expected normalized_shape
        if x.shape[-len(self.normalized_shape):] != self.normalized_shape:
            raise ValueError(f"Expected last dimensions {self.normalized_shape}, got {x.shape[-len(self.normalized_shape):]}")
        
        return triton_layernorm(x, self.weight, self.bias, self.eps)