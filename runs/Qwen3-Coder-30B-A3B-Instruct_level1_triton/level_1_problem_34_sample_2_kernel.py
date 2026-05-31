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
    total_elements = height * width
    
    # Initialize accumulators
    sum_x = tl.zeros((1,), dtype=tl.float32)
    sum_x2 = tl.zeros((1,), dtype=tl.float32)
    
    # Compute mean and variance
    for i in range(0, total_elements, BLOCK_SIZE):
        # Calculate global offset
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < total_elements
        
        # Load data
        x_offsets = batch_idx * channels * height * width + channel_idx * height * width + offset
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # Accumulate sum and sum of squares
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)
    
    # Store partial sums in shared memory
    tid = tl.thread_id()
    shared_mean[tid] = sum_x
    shared_var[tid] = sum_x2
    tl.sync()
    
    # Reduce across threads in block
    if tid == 0:
        mean_val = tl.sum(shared_mean[:BLOCK_SIZE]) / total_elements
        var_val = tl.sum(shared_var[:BLOCK_SIZE]) / total_elements - mean_val * mean_val
        
        # Store mean and variance
        mean_offset = batch_idx * channels + channel_idx
        var_offset = batch_idx * channels + channel_idx
        tl.store(mean_ptr + mean_offset, mean_val)
        tl.store(var_ptr + var_offset, var_val)
    
    tl.sync()
    
    # Load mean and variance
    mean_offset = batch_idx * channels + channel_idx
    var_offset = batch_idx * channels + channel_idx
    mean_val = tl.load(mean_ptr + mean_offset)
    var_val = tl.load(var_ptr + var_offset)
    
    # Normalize and apply affine transformation
    for i in range(0, total_elements, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < total_elements
        
        # Calculate global offset
        x_offsets = batch_idx * channels * height * width + channel_idx * height * width + offset
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # Normalize
        normalized = (x - mean_val) / tl.sqrt(var_val + eps)
        
        # Apply affine transformation
        weight = tl.load(weight_ptr + channel_idx)
        bias = tl.load(bias_ptr + channel_idx)
        out = normalized * weight + bias
        
        # Store result
        tl.store(out_ptr + x_offsets, out, mask=mask)

def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of Instance Normalization
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert weight.is_cuda and bias.is_cuda, "Weight and bias tensors must be on CUDA."
    
    batch_size, channels, height, width = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Prepare mean and variance tensors
    mean = torch.empty(batch_size * channels, dtype=torch.float32, device=x.device)
    var = torch.empty(batch_size * channels, dtype=torch.float32, device=x.device)
    
    # Grid configuration
    grid = (
        batch_size,
        channels,
    )
    
    # Launch kernel
    BLOCK_SIZE = 1024
    instance_norm_kernel[grid](
        x,
        weight,
        bias,
        out,
        mean,
        var,
        batch_size,
        channels,
        height,
        width,
        eps,
        BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for Instance Normalization.
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