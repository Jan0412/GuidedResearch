import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def batch_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the row index (channel dimension)
    row_idx = tl.program_id(0)
    
    if row_idx >= C:
        return
        
    # Calculate offset for this channel
    channel_offset = row_idx * H * W
    
    # Shared memory for mean and variance calculation
    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load data for this channel
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * H * W
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Accumulate mean
        mean += x_vals
        
    # Compute mean for this channel
    mean = mean / (H * W)
    
    # Compute variance
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * H * W
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        var += (x_vals - mean) * (x_vals - mean)
        
    var = var / (H * W)
    
    # Store mean and variance
    tl.store(mean_ptr + row_idx, mean)
    tl.store(var_ptr + row_idx, var)
    
    # Normalize and apply affine transformation
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias for this channel
    weight = tl.load(weight_ptr + row_idx, mask=row_idx < C, other=1.0)
    bias = tl.load(bias_ptr + row_idx, mask=row_idx < C, other=0.0)
    
    # Apply normalization and affine transformation
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * H * W
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        normalized = (x_vals - mean) * inv_std
        output_val = normalized * weight + bias
        tl.store(output_ptr + offsets, output_val, mask=mask)

@triton.jit
def batch_norm_forward_fused_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the row index (channel dimension)
    row_idx = tl.program_id(0)
    
    if row_idx >= C:
        return
        
    # Calculate offset for this channel
    channel_offset = row_idx * H * W
    
    # Compute mean and variance in one pass
    mean = 0.0
    var = 0.0
    
    # First pass: compute mean
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * H * W
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        mean += tl.sum(x_vals)
        
    mean = mean / (H * W)
    
    # Second pass: compute variance
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * H * W
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        diff = x_vals - mean
        var += tl.sum(diff * diff)
        
    var = var / (H * W)
    
    # Store mean and variance
    tl.store(mean_ptr + row_idx, mean)
    tl.store(var_ptr + row_idx, var)
    
    # Apply normalization and affine transformation
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias for this channel
    weight = tl.load(weight_ptr + row_idx, mask=row_idx < C, other=1.0)
    bias = tl.load(bias_ptr + row_idx, mask=row_idx < C, other=0.0)
    
    # Apply normalization and affine transformation
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * H * W
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        normalized = (x_vals - mean) * inv_std
        output_val = normalized * weight + bias
        tl.store(output_ptr + offsets, output_val, mask=mask)

def triton_batch_norm_forward(x, weight, bias, running_mean, running_var, eps=1e-5):
    """
    Triton implementation of batch normalization forward pass
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    assert weight.is_cuda and bias.is_cuda, "Weight and bias must be on CUDA."
    
    N, C, H, W = x.shape
    x = x.contiguous()
    output = torch.empty_like(x)
    
    # Allocate memory for intermediate results
    mean = torch.empty(C, dtype=torch.float32, device=x.device)
    var = torch.empty(C, dtype=torch.float32, device=x.device)
    
    # Kernel configuration
    BLOCK_SIZE = 1024
    grid = (C,)
    
    # Launch kernel
    batch_norm_forward_fused_kernel[grid](
        x, weight, bias, mean, var, output,
        N, C, H, W, eps, BLOCK_SIZE
    )
    
    return output, mean, var

class ModelNew(nn.Module):
    """
    Optimized model with custom Triton kernels for Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer with custom Triton kernel.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Batch Normalization using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        # Use our Triton implementation
        output, mean, var = triton_batch_norm_forward(
            x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
        )
        
        # Update running statistics
        with torch.no_grad():
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
        return output

# Helper functions for the kernel
def _get_block_size():
    """Get optimal block size for Triton kernel"""
    return 1024

def _get_grid_size(channel_count):
    """Calculate grid size for Triton kernel"""
    return (channel_count,)