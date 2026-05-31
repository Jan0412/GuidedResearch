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
    # Get the block index
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Shared memory for reduction operations
    shared_mean = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    shared_var = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Calculate the starting position for this thread
    start_pos = batch_idx * channels * height * width + channel_idx * height * width
    
    # Load data
    x_block = tl.load(x_ptr + start_pos + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < height * width)
    
    # Compute mean
    mean = tl.sum(x_block) / (height * width)
    tl.store(mean_ptr + batch_idx * channels + channel_idx, mean)
    
    # Compute variance
    diff = x_block - mean
    var = tl.sum(diff * diff) / (height * width)
    tl.store(var_ptr + batch_idx * channels + channel_idx, var)
    
    # Normalize and apply affine transformation
    x_normalized = (x_block - mean) / tl.sqrt(var + eps)
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    out = x_normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + start_pos + tl.arange(0, BLOCK_SIZE), out, mask=tl.arange(0, BLOCK_SIZE) < height * width)

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
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block index
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate the starting position for this thread
    start_pos = batch_idx * channels * height * width + channel_idx * height * width
    
    # Load data
    x_block = tl.load(x_ptr + start_pos + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < height * width)
    
    # Compute mean and variance in one pass
    mean = tl.sum(x_block) / (height * width)
    diff = x_block - mean
    var = tl.sum(diff * diff) / (height * width)
    
    # Normalize and apply affine transformation
    x_normalized = (x_block - mean) / tl.sqrt(var + eps)
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    out = x_normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + start_pos + tl.arange(0, BLOCK_SIZE), out, mask=tl.arange(0, BLOCK_SIZE) < height * width)

def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of Instance Normalization
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert weight.is_cuda and bias.is_cuda, "Weight and bias tensors must be on CUDA."
    
    batch_size, channels, height, width = x.shape
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Allocate memory for intermediate results
    mean = torch.empty((batch_size, channels), dtype=torch.float32, device='cuda')
    var = torch.empty((batch_size, channels), dtype=torch.float32, device='cuda')
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (batch_size, channels)
    
    # Launch kernel
    instance_norm_fused_kernel[grid](
        x,
        weight,
        bias,
        out,
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
    Optimized Model using Triton kernels for Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer with optimized Triton kernel.

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
        Applies optimized Instance Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x, self.weight, self.bias, self.eps)