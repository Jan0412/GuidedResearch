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
    rstd_ptr,
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
    shared_mean = tl.shared_ptr(mean_ptr, BLOCK_SIZE)
    shared_var = tl.shared_ptr(rstd_ptr, BLOCK_SIZE)
    
    # Calculate total elements per batch
    elements_per_batch = height * width
    
    # Initialize mean and variance
    mean = 0.0
    var = 0.0
    
    # Compute mean
    for i in range(0, elements_per_batch, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < elements_per_batch
        
        # Load data
        x_data = tl.load(x_ptr + batch_idx * channels * height * width + 
                        channel_idx * height * width + offset, mask=mask, other=0.0)
        
        # Accumulate sum
        mean += tl.sum(x_data)
    
    # Reduce across all elements in batch
    mean = tl.sum(mean) / (elements_per_batch)
    
    # Compute variance
    for i in range(0, elements_per_batch, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < elements_per_batch
        
        # Load data
        x_data = tl.load(x_ptr + batch_idx * channels * height * width + 
                        channel_idx * height * width + offset, mask=mask, other=0.0)
        
        # Compute squared differences from mean
        diff = x_data - mean
        var += tl.sum(diff * diff)
    
    # Reduce across all elements in batch
    var = tl.sum(var) / (elements_per_batch)
    
    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and rstd for this batch/channel
    if tl.thread_id() == 0:
        mean_ptr[batch_idx * channels + channel_idx] = mean
        rstd_ptr[batch_idx * channels + channel_idx] = rstd
    
    # Synchronize threads
    tl.sync()
    
    # Normalize and apply affine transformation
    for i in range(0, elements_per_batch, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < elements_per_batch
        
        # Load data
        x_data = tl.load(x_ptr + batch_idx * channels * height * width + 
                        channel_idx * height * width + offset, mask=mask, other=0.0)
        
        # Get mean and rstd
        mean_val = tl.load(mean_ptr + batch_idx * channels + channel_idx)
        rstd_val = tl.load(rstd_ptr + batch_idx * channels + channel_idx)
        
        # Normalize
        normalized = (x_data - mean_val) * rstd_val
        
        # Apply affine transformation
        weight_val = tl.load(weight_ptr + channel_idx)
        bias_val = tl.load(bias_ptr + channel_idx)
        out_data = normalized * weight_val + bias_val
        
        # Store result
        tl.store(out_ptr + batch_idx * channels * height * width + 
                channel_idx * height * width + offset, out_data, mask=mask)

# Optimized version using better reduction patterns
@triton.jit
def instance_norm_kernel_optimized(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    channels,
    height,
    width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate total elements per batch
    elements_per_batch = height * width
    
    # Shared memory for reductions
    block_mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    block_var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute mean
    mean = 0.0
    for i in range(0, elements_per_batch, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < elements_per_batch
        
        # Load data
        x_data = tl.load(x_ptr + batch_idx * channels * height * width + 
                        channel_idx * height * width + offset, mask=mask, other=0.0)
        
        # Accumulate sum
        block_mean += x_data
        mean += tl.sum(x_data)
    
    # Final mean computation
    mean = mean / elements_per_batch
    
    # Compute variance
    var = 0.0
    for i in range(0, elements_per_batch, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < elements_per_batch
        
        # Load data
        x_data = tl.load(x_ptr + batch_idx * channels * height * width + 
                        channel_idx * height * width + offset, mask=mask, other=0.0)
        
        # Compute squared differences from mean
        diff = x_data - mean
        var += tl.sum(diff * diff)
    
    # Final variance computation
    var = var / elements_per_batch
    
    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and rstd for this batch/channel
    if tl.thread_id() == 0:
        mean_ptr[batch_idx * channels + channel_idx] = mean
        rstd_ptr[batch_idx * channels + channel_idx] = rstd
    
    # Synchronize threads
    tl.sync()
    
    # Normalize and apply affine transformation
    for i in range(0, elements_per_batch, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < elements_per_batch
        
        # Load data
        x_data = tl.load(x_ptr + batch_idx * channels * height * width + 
                        channel_idx * height * width + offset, mask=mask, other=0.0)
        
        # Get mean and rstd
        mean_val = tl.load(mean_ptr + batch_idx * channels + channel_idx)
        rstd_val = tl.load(rstd_ptr + batch_idx * channels + channel_idx)
        
        # Normalize
        normalized = (x_data - mean_val) * rstd_val
        
        # Apply affine transformation
        weight_val = tl.load(weight_ptr + channel_idx)
        bias_val = tl.load(bias_ptr + channel_idx)
        out_data = normalized * weight_val + bias_val
        
        # Store result
        tl.store(out_ptr + batch_idx * channels * height * width + 
                channel_idx * height * width + offset, out_data, mask=mask)

def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                        eps: float = 1e-5):
    """
    Triton implementation of Instance Normalization for 2D tensors.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert weight.is_cuda and bias.is_cuda, "Weight and bias tensors must be on CUDA."
    
    batch_size, channels, height, width = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Allocate memory for intermediate results
    mean = torch.empty(batch_size * channels, dtype=torch.float32, device='cuda')
    rstd = torch.empty(batch_size * channels, dtype=torch.float32, device='cuda')
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024
    grid = (batch_size, channels)
    
    # Launch kernel
    instance_norm_kernel_optimized[grid](
        x, weight, bias, out, mean, rstd,
        batch_size, channels, height, width, eps,
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