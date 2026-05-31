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
    # Calculate global thread index
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Each thread processes one element
    if idx < batch_size * channels * height * width:
        # Convert linear index to multi-dimensional indices
        batch_idx = idx // (channels * height * width)
        remaining = idx % (channels * height * width)
        channel_idx = remaining // (height * width)
        remaining = remaining % (height * width)
        h_idx = remaining // width
        w_idx = remaining % width
        
        # Get current channel
        channel_offset = channel_idx * height * width
        x_offset = batch_idx * channels * height * width + channel_offset
        weight_offset = channel_idx
        bias_offset = channel_idx
        
        # Compute mean for this channel across batch and spatial dimensions
        mean = 0.0
        for i in range(height * width):
            mean += tl.load(x_ptr + x_offset + i)
        mean /= (height * width)
        
        # Store mean for this channel
        mean_offset = batch_idx * channels + channel_idx
        tl.store(mean_ptr + mean_offset, mean)
        
        # Compute variance
        var = 0.0
        for i in range(height * width):
            val = tl.load(x_ptr + x_offset + i)
            diff = val - mean
            var += diff * diff
        var /= (height * width)
        rstd = 1.0 / tl.sqrt(var + eps)
        
        # Store reciprocal standard deviation
        tl.store(rstd_ptr + mean_offset, rstd)
        
        # Normalize and apply affine transformation
        x_val = tl.load(x_ptr + idx)
        normalized = (x_val - mean) * rstd
        weight_val = tl.load(weight_ptr + weight_offset)
        bias_val = tl.load(bias_ptr + bias_offset)
        out_val = normalized * weight_val + bias_val
        tl.store(out_ptr + idx, out_val)

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
    # Calculate global thread index
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Each thread processes one element
    if idx < batch_size * channels * height * width:
        # Convert linear index to multi-dimensional indices
        batch_idx = idx // (channels * height * width)
        remaining = idx % (channels * height * width)
        channel_idx = remaining // (height * width)
        remaining = remaining % (height * width)
        h_idx = remaining // width
        w_idx = remaining % width
        
        # Get current channel data
        channel_offset = channel_idx * height * width
        x_offset = batch_idx * channels * height * width + channel_offset
        
        # Compute mean for this channel
        mean = 0.0
        for i in range(height * width):
            mean += tl.load(x_ptr + x_offset + i)
        mean /= (height * width)
        
        # Compute variance
        var = 0.0
        for i in range(height * width):
            val = tl.load(x_ptr + x_offset + i)
            diff = val - mean
            var += diff * diff
        var /= (height * width)
        rstd = 1.0 / tl.sqrt(var + eps)
        
        # Normalize and apply affine transformation
        x_val = tl.load(x_ptr + idx)
        normalized = (x_val - mean) * rstd
        weight_val = tl.load(weight_ptr + channel_idx)
        bias_val = tl.load(bias_ptr + channel_idx)
        out_val = normalized * weight_val + bias_val
        tl.store(out_ptr + idx, out_val)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Instance Normalization.
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
        Applies optimized Instance Normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        batch_size, channels, height, width = x.shape
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Use fused kernel for better performance
        BLOCK_SIZE = 1024
        grid = lambda meta: ((batch_size * channels * height * width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        
        # Launch Triton kernel
        instance_norm_fused_kernel[grid](
            x, 
            self.weight, 
            self.bias, 
            out,
            batch_size,
            channels,
            height,
            width,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out