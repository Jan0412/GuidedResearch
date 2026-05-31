import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def instance_norm_kernel(
    X_ptr,  # Input tensor pointer (B, C, H, W)
    Y_ptr,  # Output tensor pointer
    weight_ptr,  # Gamma (scale) pointer
    bias_ptr,  # Beta (shift) pointer
    mean_ptr,  # Mean pointer (B, C)
    var_ptr,  # Variance pointer (B, C)
    N_elements,  # Total number of elements in X
    C,  # Number of channels
    H, W,  # Spatial dimensions
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one (batch, channel) pair
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # channel index
    
    # Compute spatial size per (batch, channel)
    spatial_size = H * W
    
    # Compute pointer offsets for this (batch, channel) pair
    offset_bc = pid_b * C * H * W + pid_c * H * W
    
    # Compute mean and variance using online algorithm
    # First pass: compute mean
    sum_val = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = (offsets < spatial_size)
        idx = offset_bc + offsets
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += x
        
    mean = sum_val / spatial_size
    
    # Second pass: compute variance
    var_sum = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = (offsets < spatial_size)
        idx = offset_bc + offsets
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        diff = x - mean
        var_sum += diff * diff
    
    var = var_sum / spatial_size
    
    # Store mean and variance for potential reuse (though we don't need them in this kernel)
    # If needed, could store to mean_ptr and var_ptr, but InstanceNorm2d doesn't require storing them
    
    # Third pass: normalize and apply scale/bias
    std = tl.sqrt(var + eps)
    
    # Get weight and bias for this channel
    w = tl.load(weight_ptr + pid_c) if weight_ptr is not None else 1.0
    b = tl.load(bias_ptr + pid_c) if bias_ptr is not None else 0.0
    
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = (offsets < spatial_size)
        idx = offset_bc + offsets
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        
        # Normalize: (x - mean) / std * weight + bias
        x_norm = (x - mean) / std * w + b
        tl.store(Y_ptr + idx, x_norm, mask=mask)


def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of Instance Normalization for 4D tensors (B, C, H, W)
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert weight.is_cuda, "Weight tensor must be on CUDA."
    assert bias.is_cuda, "Bias tensor must be on CUDA."
    
    x = x.contiguous()
    weight = weight.contiguous() if weight is not None else None
    bias = bias.contiguous() if bias is not None else None
    
    # Get dimensions
    B, C, H, W = x.shape
    spatial_size = H * W
    
    # Prepare output tensor
    y = torch.empty_like(x)
    
    # Configure grid: (B, C) - one program per (batch, channel) pair
    grid = (B, C)
    
    # Block size for spatial dimension
    BLOCK_SIZE = 256
    
    # Launch kernel
    instance_norm_kernel[grid](
        x, y, weight, bias, None, None,  # mean and variance pointers (not stored)
        x.numel(), C, H, W, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using Triton kernel.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer with Triton kernel implementation.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        # Initialize weight and bias parameters to match nn.InstanceNorm2d defaults
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization using Triton kernel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x, self.weight, self.bias, self.eps)


# Reuse the provided functions
batch_size = 112  # heavier workload
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]