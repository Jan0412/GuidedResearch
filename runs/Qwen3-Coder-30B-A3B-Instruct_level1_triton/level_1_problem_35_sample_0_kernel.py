import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def group_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    channels,
    spatial_size,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    
    # Calculate channel range for this group
    channels_per_group = channels // num_groups
    start_channel = group_id * channels_per_group
    end_channel = start_channel + channels_per_group
    
    # Shared memory for reduction
    shared_mean = tl.shared_ptr(tl.float32, BLOCK_SIZE)
    shared_var = tl.shared_ptr(tl.float32, BLOCK_SIZE)
    
    # Initialize accumulators
    sum_x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over spatial dimensions
    for i in range(0, spatial_size, BLOCK_SIZE):
        # Calculate offsets
        offset = batch_id * channels * spatial_size + start_channel * spatial_size + i + tl.arange(0, BLOCK_SIZE)
        mask = offset < batch_size * channels * spatial_size
        
        # Load data
        x_vals = tl.load(x_ptr + offset, mask=mask, other=0.0)
        
        # Accumulate sum and sum of squares
        sum_x += x_vals
        sum_x2 += x_vals * x_vals
    
    # Reduce within block
    mean_block = tl.sum(sum_x) / (spatial_size * channels_per_group)
    var_block = tl.sum(sum_x2) / (spatial_size * channels_per_group) - mean_block * mean_block
    
    # Store block results
    tl.store(mean_ptr + batch_id * num_groups + group_id, mean_block)
    tl.store(rstd_ptr + batch_id * num_groups + group_id, 1.0 / tl.sqrt(var_block + eps))
    
    # Compute normalized values
    for i in range(0, spatial_size, BLOCK_SIZE):
        # Calculate offsets
        offset = batch_id * channels * spatial_size + start_channel * spatial_size + i + tl.arange(0, BLOCK_SIZE)
        mask = offset < batch_size * channels * spatial_size
        
        # Load data
        x_vals = tl.load(x_ptr + offset, mask=mask, other=0.0)
        
        # Normalize
        mean_val = tl.load(mean_ptr + batch_id * num_groups + group_id)
        rstd_val = tl.load(rstd_ptr + batch_id * num_groups + group_id)
        normalized = (x_vals - mean_val) * rstd_val
        
        # Apply scale and shift
        channel_idx = (offset % (channels * spatial_size)) // spatial_size
        weight_vals = tl.load(weight_ptr + channel_idx, mask=channel_idx < channels, other=1.0)
        bias_vals = tl.load(bias_ptr + channel_idx, mask=channel_idx < channels, other=0.0)
        
        # Apply affine transformation
        out_vals = normalized * weight_vals + bias_vals
        
        # Store output
        tl.store(out_ptr + offset, out_vals, mask=mask)


@triton.jit
def group_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    channels,
    spatial_size,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    
    # Calculate channel range for this group
    channels_per_group = channels // num_groups
    start_channel = group_id * channels_per_group
    end_channel = start_channel + channels_per_group
    
    # Load mean and rstd for this batch and group
    mean_val = tl.load(mean_ptr + batch_id * num_groups + group_id)
    rstd_val = tl.load(rstd_ptr + batch_id * num_groups + group_id)
    
    # Process elements in chunks
    for i in range(0, spatial_size, BLOCK_SIZE):
        # Calculate offsets
        offset = batch_id * channels * spatial_size + start_channel * spatial_size + i + tl.arange(0, BLOCK_SIZE)
        mask = offset < batch_size * channels * spatial_size
        
        # Load data
        x_vals = tl.load(x_ptr + offset, mask=mask, other=0.0)
        
        # Normalize
        normalized = (x_vals - mean_val) * rstd_val
        
        # Apply scale and shift
        channel_idx = (offset % (channels * spatial_size)) // spatial_size
        weight_vals = tl.load(weight_ptr + channel_idx, mask=channel_idx < channels, other=1.0)
        bias_vals = tl.load(bias_ptr + channel_idx, mask=channel_idx < channels, other=0.0)
        
        # Apply affine transformation
        out_vals = normalized * weight_vals + bias_vals
        
        # Store output
        tl.store(out_ptr + offset, out_vals, mask=mask)


def triton_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                     num_groups: int, eps: float = 1e-5):
    """
    Triton implementation of GroupNorm with fused operations
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda and bias.is_cuda, "Weight and bias must be on CUDA"
    
    batch_size, channels, *spatial_dims = x.shape
    spatial_size = torch.prod(torch.tensor(spatial_dims)).item()
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Allocate intermediate tensors for mean and rstd
    mean = torch.empty((batch_size, num_groups), dtype=torch.float32, device=x.device)
    rstd = torch.empty((batch_size, num_groups), dtype=torch.float32, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024
    grid_0 = (batch_size, num_groups)
    
    # Launch kernel to compute mean and rstd
    group_norm_kernel[grid_0](
        x, weight, bias, out, mean, rstd,
        batch_size, channels, spatial_size, num_groups, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out, mean, rstd


class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernels for Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = 1e-5
        
        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Use our optimized Triton implementation
        out, _, _ = triton_group_norm(x, self.weight, self.bias, self.num_groups, self.eps)
        return out