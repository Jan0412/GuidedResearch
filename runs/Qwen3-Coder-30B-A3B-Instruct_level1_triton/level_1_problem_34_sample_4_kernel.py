import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def instance_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    batch_size,
    channels,
    height,
    width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Shared memory for reduction
    shared_mean = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    shared_var = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Calculate offset for this batch and channel
    batch_offset = batch_idx * channels * height * width
    channel_offset = channel_idx * height * width
    
    # Load weights and biases
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Compute mean and variance
    mean = 0.0
    var = 0.0
    
    # First pass: compute mean
    for i in range(0, height * width, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < height * width
        
        # Load data
        x_data = tl.load(x_ptr + batch_offset + channel_offset + idx, mask=mask, other=0.0)
        
        # Accumulate sum for mean
        mean += tl.sum(x_data)
    
    # Reduce across threads
    mean = tl.sum(mean) / (height * width)
    
    # Store mean in shared memory
    tl.store(mean_ptr + batch_idx * channels + channel_idx, mean)
    
    # Second pass: compute variance
    var = 0.0
    for i in range(0, height * width, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < height * width
        
        # Load data
        x_data = tl.load(x_ptr + batch_offset + channel_offset + idx, mask=mask, other=0.0)
        
        # Compute squared differences
        diff = x_data - mean
        var += tl.sum(diff * diff)
    
    # Reduce across threads
    var = tl.sum(var) / (height * width)
    
    # Store variance in shared memory
    tl.store(var_ptr + batch_idx * channels + channel_idx, var)
    
    # Normalize and apply affine transformation
    for i in range(0, height * width, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < height * width
        
        # Load data
        x_data = tl.load(x_ptr + batch_offset + channel_offset + idx, mask=mask, other=0.0)
        
        # Normalize
        std = tl.sqrt(var + eps)
        normalized = (x_data - mean) / std
        
        # Apply affine transformation
        result = normalized * weight + bias
        
        # Store result
        tl.store(out_ptr + batch_offset + channel_offset + idx, result, mask=mask)

@triton.jit
def instance_norm_fused_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate offset for this batch and channel
    batch_offset = batch_idx * channels * height * width
    channel_offset = channel_idx * height * width
    
    # Load weights and biases
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Compute mean and variance in one pass
    mean = 0.0
    sum_sq = 0.0
    
    # Process all elements for this channel
    for i in range(0, height * width, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < height * width
        
        # Load data
        x_data = tl.load(x_ptr + batch_offset + channel_offset + idx, mask=mask, other=0.0)
        
        # Accumulate mean and sum of squares
        mean += tl.sum(x_data)
        sum_sq += tl.sum(x_data * x_data)
    
    # Reduce across threads
    mean = tl.sum(mean) / (height * width)
    sum_sq = tl.sum(sum_sq) / (height * width)
    
    # Compute variance from mean and sum of squares
    var = sum_sq - mean * mean
    
    # Normalize and apply affine transformation
    for i in range(0, height * width, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < height * width
        
        # Load data
        x_data = tl.load(x_ptr + batch_offset + channel_offset + idx, mask=mask, other=0.0)
        
        # Normalize
        std = tl.sqrt(var + eps)
        normalized = (x_data - mean) / std
        
        # Apply affine transformation
        result = normalized * weight + bias
        
        # Store result
        tl.store(out_ptr + batch_offset + channel_offset + idx, result, mask=mask)

def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of Instance Normalization
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert weight.is_cuda and bias.is_cuda, "Weight and bias tensors must be on CUDA."
    
    batch_size, channels, height, width = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Allocate memory for intermediate results (mean and variance)
    mean = torch.empty((batch_size, channels), dtype=torch.float32, device='cuda')
    var = torch.empty((batch_size, channels), dtype=torch.float32, device='cuda')
    
    # Set up grid
    grid = (batch_size, channels)
    BLOCK_SIZE = 1024
    
    # Launch kernel
    instance_norm_fused_kernel[grid](
        x, weight, bias, out,
        batch_size, channels, height, width,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x, self.weight, self.bias, self.eps)