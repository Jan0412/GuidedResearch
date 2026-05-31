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
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Shared memory for reduction
    shared_mean = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    shared_var = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Calculate total elements per batch/channel
    elements_per_batch_channel = height * width
    
    # Initialize accumulators
    sum_x = tl.zeros((1,), dtype=tl.float32)
    sum_x2 = tl.zeros((1,), dtype=tl.float32)
    
    # Reduction to compute mean and variance
    for i in range(0, elements_per_batch_channel, BLOCK_SIZE):
        # Calculate global offset
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < elements_per_batch_channel
        
        # Calculate actual index in flattened tensor
        batch_offset = batch_idx * channels * height * width
        channel_offset = channel_idx * height * width
        global_offset = batch_offset + channel_offset + offset
        
        # Load data
        x = tl.load(x_ptr + global_offset, mask=mask, other=0.0)
        
        # Accumulate sum and sum of squares
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)
    
    # Compute mean and variance
    mean = sum_x / elements_per_batch_channel
    var = sum_x2 / elements_per_batch_channel - mean * mean
    
    # Store mean and variance
    if channel_idx < channels:
        tl.store(mean_ptr + batch_idx * channels + channel_idx, mean)
        tl.store(var_ptr + batch_idx * channels + channel_idx, var)
    
    # Synchronize threads
    tl.sync()
    
    # Normalize and apply affine transformation
    for i in range(0, elements_per_batch_channel, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < elements_per_batch_channel
        
        batch_offset = batch_idx * channels * height * width
        channel_offset = channel_idx * height * width
        global_offset = batch_offset + channel_offset + offset
        
        # Load data
        x = tl.load(x_ptr + global_offset, mask=mask, other=0.0)
        
        # Normalize
        x_normalized = (x - mean) / tl.sqrt(var + eps)
        
        # Apply affine transformation
        weight = tl.load(weight_ptr + channel_idx, mask=True, other=1.0)
        bias = tl.load(bias_ptr + channel_idx, mask=True, other=0.0)
        out = x_normalized * weight + bias
        
        # Store result
        tl.store(out_ptr + global_offset, out, mask=mask)

@triton.jit
def instance_norm_forward_kernel(
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
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate total elements per batch/channel
    elements_per_batch_channel = height * width
    
    # Compute mean and variance using shared memory approach
    mean = tl.load(mean_ptr + batch_idx * channels + channel_idx)
    var = tl.load(var_ptr + batch_idx * channels + channel_idx)
    
    # Normalize and apply affine transformation
    for i in range(0, elements_per_batch_channel, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < elements_per_batch_channel
        
        batch_offset = batch_idx * channels * height * width
        channel_offset = channel_idx * height * width
        global_offset = batch_offset + channel_offset + offset
        
        # Load data
        x = tl.load(x_ptr + global_offset, mask=mask, other=0.0)
        
        # Normalize
        x_normalized = (x - mean) / tl.sqrt(var + eps)
        
        # Apply affine transformation
        weight = tl.load(weight_ptr + channel_idx, mask=True, other=1.0)
        bias = tl.load(bias_ptr + channel_idx, mask=True, other=0.0)
        out = x_normalized * weight + bias
        
        # Store result
        tl.store(out_ptr + global_offset, out, mask=mask)

def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                        eps: float = 1e-5):
    """
    Triton implementation of Instance Normalization
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert weight.is_cuda and bias.is_cuda, "Weight and bias tensors must be on CUDA."
    
    batch_size, channels, height, width = x.shape
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Allocate mean and variance tensors
    mean = torch.empty(batch_size, channels, device=x.device, dtype=torch.float32)
    var = torch.empty(batch_size, channels, device=x.device, dtype=torch.float32)
    
    # Use appropriate block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid1 = (batch_size, channels)
    grid2 = (batch_size, channels)
    
    # First pass: compute mean and variance
    instance_norm_kernel[grid1](
        x, weight, bias, out, mean, var,
        batch_size, channels, height, width,
        eps, BLOCK_SIZE
    )
    
    # Second pass: normalize and apply affine transformation
    instance_norm_forward_kernel[grid2](
        x, weight, bias, out, mean, var,
        batch_size, channels, height, width,
        eps, BLOCK_SIZE
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
        Applies Instance Normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        # Convert to FP32 if needed
        if x.dtype != torch.float32:
            original_dtype = x.dtype
            x = x.to(torch.float32)
            weight = self.weight.to(torch.float32)
            bias = self.bias.to(torch.float32)
        else:
            weight = self.weight
            bias = self.bias
            
        # Apply custom Triton kernel
        result = triton_instance_norm(x, weight, bias, self.eps)
        
        # Convert back to original dtype if needed
        if 'original_dtype' in locals():
            result = result.to(original_dtype)
            
        return result