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
    # Calculate global thread index
    pid = tl.program_id(0)
    group_id = pid // (batch_size * spatial_size)
    batch_and_spatial_id = pid % (batch_size * spatial_size)
    
    # Shared memory for reduction operations
    mean_shared = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    var_shared = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Each thread processes one element along the channel dimension
    channel_offset = group_id * (channels // num_groups)
    
    # Load data
    x_ptrs = x_ptr + batch_and_spatial_id * channels + channel_offset + tl.arange(0, BLOCK_SIZE)
    weight_ptrs = weight_ptr + channel_offset + tl.arange(0, BLOCK_SIZE)
    bias_ptrs = bias_ptr + channel_offset + tl.arange(0, BLOCK_SIZE)
    
    # Check bounds
    mask = (channel_offset + tl.arange(0, BLOCK_SIZE)) < (channel_offset + (channels // num_groups))
    
    # Load input data
    x_data = tl.load(x_ptrs, mask=mask, other=0.0)
    weight_data = tl.load(weight_ptrs, mask=mask, other=0.0)
    bias_data = tl.load(bias_ptrs, mask=mask, other=0.0)
    
    # Compute mean and variance
    mean = tl.sum(x_data) / (channels // num_groups)
    var = tl.sum((x_data - mean) * (x_data - mean)) / (channels // num_groups)
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply affine transformation
    normalized = (x_data - mean) * rstd
    output = normalized * weight_data + bias_data
    
    # Store results
    out_ptrs = out_ptr + batch_and_spatial_id * channels + channel_offset + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptrs, output, mask=mask)
    
    # Store mean and rstd for later use if needed
    if group_id == 0:
        mean_ptrs = mean_ptr + batch_and_spatial_id * num_groups + tl.arange(0, num_groups)
        rstd_ptrs = rstd_ptr + batch_and_spatial_id * num_groups + tl.arange(0, num_groups)
        
        # We can't store directly because each thread is working on different channels
        # But for now we'll just compute it in the first group for simplicity


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
    # Get thread id and group id
    tid = tl.program_id(0)
    group_id = tid // (batch_size * spatial_size)
    batch_spatial_idx = tid % (batch_size * spatial_size)
    
    # Compute channel offset for this group
    channels_per_group = channels // num_groups
    channel_offset = group_id * channels_per_group
    
    # Process channels in chunks
    for i in range(0, channels_per_group, BLOCK_SIZE):
        # Calculate actual channel indices
        channel_indices = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = channel_indices < (channel_offset + channels_per_group)
        
        # Load input data
        x_data = tl.load(x_ptr + batch_spatial_idx * channels + channel_indices, mask=mask, other=0.0)
        
        # Load weight and bias
        weight_data = tl.load(weight_ptr + channel_indices, mask=mask, other=0.0)
        bias_data = tl.load(bias_ptr + channel_indices, mask=mask, other=0.0)
        
        # Compute statistics for this chunk
        mean = tl.sum(x_data) / channels_per_group
        var = tl.sum((x_data - mean) * (x_data - mean)) / channels_per_group
        rstd = 1.0 / tl.sqrt(var + eps)
        
        # Normalize and scale
        normalized = (x_data - mean) * rstd
        output = normalized * weight_data + bias_data
        
        # Store output
        out_data = out_ptr + batch_spatial_idx * channels + channel_indices
        tl.store(out_data, output, mask=mask)


# Optimized version using shared memory and better reduction
@triton.jit
def group_norm_optimized_kernel(
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
    # Thread and group information
    tid = tl.program_id(0)
    group_id = tid // (batch_size * spatial_size)
    batch_spatial_idx = tid % (batch_size * spatial_size)
    
    # Channel information
    channels_per_group = channels // num_groups
    channel_offset = group_id * channels_per_group
    
    # Process all channels for this group in chunks
    for chunk_start in range(0, channels_per_group, BLOCK_SIZE):
        # Compute actual channel indices for this chunk
        channel_indices = channel_offset + chunk_start + tl.arange(0, BLOCK_SIZE)
        mask = channel_indices < (channel_offset + channels_per_group)
        
        # Load data for this chunk
        x_chunk = tl.load(x_ptr + batch_spatial_idx * channels + channel_indices, mask=mask, other=0.0)
        
        # Load weight and bias for this chunk
        weight_chunk = tl.load(weight_ptr + channel_indices, mask=mask, other=0.0)
        bias_chunk = tl.load(bias_ptr + channel_indices, mask=mask, other=0.0)
        
        # Compute mean for this chunk
        mean = tl.sum(x_chunk) / channels_per_group
        # Compute variance for this chunk
        var = tl.sum((x_chunk - mean) * (x_chunk - mean)) / channels_per_group
        rstd = 1.0 / tl.sqrt(var + eps)
        
        # Normalize and scale
        normalized = (x_chunk - mean) * rstd
        output = normalized * weight_chunk + bias_chunk
        
        # Store result
        out_chunk = out_ptr + batch_spatial_idx * channels + channel_indices
        tl.store(out_chunk, output, mask=mask)


def triton_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                      num_groups: int, eps: float = 1e-5):
    """
    Triton implementation of Group Normalization
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda and bias.is_cuda, "Weight and bias tensors must be on CUDA"
    
    batch_size, channels, *spatial_dims = x.shape
    spatial_size = math.prod(spatial_dims)
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Prepare mean and rstd tensors if needed
    mean = torch.empty(batch_size * spatial_size, num_groups, dtype=torch.float32, device=x.device)
    rstd = torch.empty(batch_size * spatial_size, num_groups, dtype=torch.float32, device=x.device)
    
    # Grid configuration
    total_threads = batch_size * spatial_size * num_groups
    BLOCK_SIZE = 128
    
    # Determine grid size
    grid_size = (total_threads + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    group_norm_optimized_kernel[grid_size](
        x, weight, bias, out, mean, rstd,
        batch_size, channels, spatial_size, num_groups, eps, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer with optimized Triton kernel.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return triton_group_norm(x, self.weight, self.bias, self.num_groups)