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
    # Get the batch and channel index for this program
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate the starting position for this thread's work
    block_start = tl.program_id(2) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total elements per batch
    elements_per_batch = channels * height * width
    
    # Calculate the base pointer for this batch and channel
    batch_offset = batch_idx * elements_per_batch
    channel_offset = channel_idx * height * width
    
    # Calculate pointers for this specific batch and channel
    x_base_ptr = x_ptr + batch_offset + channel_offset
    out_base_ptr = out_ptr + batch_offset + channel_offset
    weight_base_ptr = weight_ptr + channel_idx
    bias_base_ptr = bias_ptr + channel_idx
    
    # Load weight and bias for this channel
    weight = tl.load(weight_base_ptr)
    bias = tl.load(bias_base_ptr)
    
    # Calculate mean and variance for this channel across spatial dimensions
    # We'll compute it in chunks to avoid memory issues
    mean = 0.0
    var = 0.0
    
    # First pass: calculate mean
    mask = offsets < height * width
    if mask.any():
        x_vals = tl.load(x_base_ptr + offsets, mask=mask, other=0.0)
        mean = tl.sum(x_vals)
    
    # Reduce across all threads in the block
    mean = tl.sum(mean, axis=0)
    mean = mean / (height * width)
    
    # Second pass: calculate variance
    var = 0.0
    if mask.any():
        x_vals = tl.load(x_base_ptr + offsets, mask=mask, other=0.0)
        diff = x_vals - mean
        var = tl.sum(diff * diff)
    
    # Reduce across all threads in the block
    var = tl.sum(var, axis=0)
    var = var / (height * width)
    
    # Calculate reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and rstd for this batch and channel
    mean_addr = mean_ptr + batch_idx * channels + channel_idx
    rstd_addr = rstd_ptr + batch_idx * channels + channel_idx
    tl.store(mean_addr, mean)
    tl.store(rstd_addr, rstd)
    
    # Normalize and apply affine transformation
    if mask.any():
        x_vals = tl.load(x_base_ptr + offsets, mask=mask, other=0.0)
        normalized = (x_vals - mean) * rstd
        output = normalized * weight + bias
        tl.store(out_base_ptr + offsets, output, mask=mask)

@triton.jit
def instance_norm_forward_kernel(
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
    # Get the batch and channel index for this program
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate total elements per batch
    elements_per_batch = channels * height * width
    
    # Calculate the base pointer for this batch and channel
    batch_offset = batch_idx * elements_per_batch
    channel_offset = channel_idx * height * width
    
    # Calculate pointers for this specific batch and channel
    x_base_ptr = x_ptr + batch_offset + channel_offset
    out_base_ptr = out_ptr + batch_offset + channel_offset
    weight_base_ptr = weight_ptr + channel_idx
    bias_base_ptr = bias_ptr + channel_idx
    
    # Load weight and bias for this channel
    weight = tl.load(weight_base_ptr)
    bias = tl.load(bias_base_ptr)
    
    # Load precomputed mean and rstd
    mean_addr = mean_ptr + batch_idx * channels + channel_idx
    rstd_addr = rstd_ptr + batch_idx * channels + channel_idx
    mean = tl.load(mean_addr)
    rstd = tl.load(rstd_addr)
    
    # Process elements in chunks
    block_start = 0
    while block_start < height * width:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < height * width
        
        if mask.any():
            x_vals = tl.load(x_base_ptr + offsets, mask=mask, other=0.0)
            normalized = (x_vals - mean) * rstd
            output = normalized * weight + bias
            tl.store(out_base_ptr + offsets, output, mask=mask)
        
        block_start += BLOCK_SIZE

def triton_instance_norm(x, weight, bias, eps=1e-5):
    """
    Triton implementation of Instance Normalization
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert weight.is_cuda, "Weight tensor must be on CUDA."
    assert bias.is_cuda, "Bias tensor must be on CUDA."
    
    batch_size, channels, height, width = x.shape
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Allocate mean and rstd tensors
    mean = torch.empty((batch_size, channels), dtype=torch.float32, device='cuda')
    rstd = torch.empty((batch_size, channels), dtype=torch.float32, device='cuda')
    
    # Launch kernel for computing mean and rstd
    BLOCK_SIZE = 1024
    grid1 = (
        batch_size,
        channels,
        (height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    )
    
    # First kernel computes mean and rstd
    instance_norm_kernel[grid1](
        x, weight, bias, out, mean, rstd,
        batch_size, channels, height, width, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second kernel applies normalization
    grid2 = (
        batch_size,
        channels,
        (height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    )
    
    instance_norm_forward_kernel[grid2](
        x, weight, bias, out, mean, rstd,
        batch_size, channels, height, width, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernels for Instance Normalization.
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