import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    X_ptr,  # Input tensor pointer (batch_size, num_features, H, W)
    Y_ptr,  # Output tensor pointer
    weight_ptr,  # Scale parameter pointer (num_features,)
    bias_ptr,    # Shift parameter pointer (num_features,)
    mean_ptr,    # Precomputed means (batch_size, num_features)
    var_ptr,     # Precomputed variances (batch_size, num_features)
    batch_size, num_features, height, width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute indices for batch and feature
    pid_b = tl.program_id(0)
    pid_f = tl.program_id(1)
    
    # Get mean and variance for this (batch, feature) pair
    mean = tl.load(mean_ptr + pid_b * num_features + pid_f)
    var = tl.load(var_ptr + pid_b * num_features + pid_f)
    
    # Compute standard deviation with numerical stability
    std = tl.sqrt(var + eps)
    
    # Load scale and bias for this feature
    w = tl.load(weight_ptr + pid_f) if weight_ptr is not None else 1.0
    b = tl.load(bias_ptr + pid_f) if bias_ptr is not None else 0.0
    
    # Compute number of elements per (batch, feature) spatial map
    spatial_size = height * width
    
    # Iterate over spatial positions in blocks
    block_start = 0
    while block_start < spatial_size:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Compute 2D indices from 1D offset
        h = offsets // width
        w_pos = offsets % width
        
        # Compute flat index for X and Y
        idx = pid_b * (num_features * spatial_size) + pid_f * spatial_size + offsets
        
        # Load input
        x = tl.load(X_ptr + idx, mask=mask)
        
        # Normalize: (x - mean) / std * weight + bias
        normalized = (x - mean) / std * w + b
        
        # Store result
        tl.store(Y_ptr + idx, normalized, mask=mask)
        
        block_start += BLOCK_SIZE


def instance_norm_forward(x, weight, bias, eps=1e-5):
    """
    Triton implementation of InstanceNorm2d forward pass.
    """
    batch_size, num_features, height, width = x.shape
    x = x.contiguous()
    
    # Compute statistics per (batch, channel) spatial map
    # Reshape to (batch_size * num_features, height * width) for easier stats computation
    x_view = x.view(batch_size * num_features, -1)
    
    # Compute mean and variance along spatial dimensions
    mean = x_view.mean(dim=1)
    var = x_view.var(dim=1, unbiased=False)
    
    # Prepare output tensor
    y = torch.empty_like(x)
    
    # Allocate pointers for weight and bias if they exist
    weight_ptr = weight.data_ptr() if weight is not None else None
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Define grid: one block per (batch, feature) pair
    grid = (batch_size, num_features)
    
    # Launch kernel
    BLOCK_SIZE = 512
    instance_norm_kernel[grid](
        x, y, weight_ptr, bias_ptr, mean, var,
        batch_size, num_features, height, width,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using custom Triton kernel.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer with optimized Triton implementation.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        # Initialize weight and bias parameters (like nn.InstanceNorm2d does)
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
        return instance_norm_forward(x, self.weight, self.bias, self.eps)


# Define model parameters matching the original
batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]