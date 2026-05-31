import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr,  # Input tensor (B, C, H, W)
    y_ptr,  # Output tensor
    gamma_ptr,  # Scale parameter (C,)
    beta_ptr,  # Shift parameter (C,)
    n_elements,  # Total number of elements
    B, C, H, W,  # Dimensions
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes one (batch, channel) pair
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate offsets for this (batch, channel)
    # Memory layout: (B, C, H, W) -> row-major
    base_offset = batch_idx * C * H * W + channel_idx * H * W
    
    # Compute mean and variance using online algorithm
    mean = 0.0
    var_sum = 0.0
    
    # Process all elements in this (batch, channel) pair
    for hw_idx in range(H * W):
        offset = base_offset + hw_idx
        if offset < n_elements:
            x_val = tl.load(x_ptr + offset)
            # Online mean update
            mean += x_val
            # For variance, we'll use the sum of squared differences from mean
            var_sum += x_val * x_val
    
    # Finalize mean
    n = H * W
    mean = mean / n
    
    # Compute variance: E[X^2] - E[X]^2
    var = var_sum / n - mean * mean
    
    # Compute scale factor
    std = tl.sqrt(var + eps)
    
    # Apply normalization and affine transformation
    gamma = tl.load(gamma_ptr + channel_idx) if gamma_ptr is not None else 1.0
    beta = tl.load(beta_ptr + channel_idx) if beta_ptr is not None else 0.0
    
    for hw_idx in range(H * W):
        offset = base_offset + hw_idx
        if offset < n_elements:
            x_val = tl.load(x_ptr + offset)
            normalized = (x_val - mean) / std
            out_val = gamma * normalized + beta
            tl.store(y_ptr + offset, out_val)


def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of InstanceNorm2d.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Scale parameter of shape (C,)
        bias: Shift parameter of shape (C,)
        eps: Small constant for numerical stability
    
    Returns:
        Output tensor of shape (B, C, H, W)
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    B, C, H, W = x.shape
    n_elements = x.numel()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Grid: (batch_size, num_channels)
    grid = (B, C)
    
    # Launch the kernel
    BLOCK_SIZE = 256  # Not actually used in this kernel since we process all elements per block
    
    instance_norm_kernel[grid](
        x, out, 
        weight if weight is not None else None, 
        bias if bias is not None else None,
        n_elements, B, C, H, W,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer with Triton optimization.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        # Register buffers for weight and bias to match nn.InstanceNorm2d behavior
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x, self.weight, self.bias, self.eps)