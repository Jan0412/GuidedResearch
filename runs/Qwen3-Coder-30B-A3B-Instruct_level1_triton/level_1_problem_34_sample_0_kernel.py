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
    channel_offset = batch_idx * channels * height * width + channel_idx * height * width
    
    # Load weights and biases
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Compute mean
    mean = 0.0
    for i in range(0, height * width, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < height * width
        x_vals = tl.load(x_ptr + channel_offset + offsets, mask=mask, other=0.0)
        mean += tl.sum(x_vals)
    
    mean = mean / (height * width)
    
    # Store mean in shared memory
    tl.store(mean_ptr + batch_idx * channels + channel_idx, mean)
    
    # Compute variance
    var = 0.0
    for i in range(0, height * width, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < height * width
        x_vals = tl.load(x_ptr + channel_offset + offsets, mask=mask, other=0.0)
        diff = x_vals - mean
        var += tl.sum(diff * diff)
    
    var = var / (height * width)
    
    # Store variance in shared memory
    tl.store(var_ptr + batch_idx * channels + channel_idx, var)
    
    # Normalize and apply affine transformation
    for i in range(0, height * width, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < height * width
        x_vals = tl.load(x_ptr + channel_offset + offsets, mask=mask, other=0.0)
        
        # Normalize
        normalized = (x_vals - mean) / tl.sqrt(var + eps)
        
        # Apply affine transformation
        out_val = normalized * weight + bias
        
        # Store result
        tl.store(out_ptr + channel_offset + offsets, out_val, mask=mask)

@triton.jit
def fused_instance_norm_matmul_kernel(
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
    
    # Calculate offset for this batch and channel
    channel_offset = batch_idx * channels * height * width + channel_idx * height * width
    
    # Load weights and biases
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Compute mean and variance in one pass
    sum_x = 0.0
    sum_x2 = 0.0
    
    for i in range(0, height * width, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < height * width
        x_vals = tl.load(x_ptr + channel_offset + offsets, mask=mask, other=0.0)
        sum_x += tl.sum(x_vals)
        sum_x2 += tl.sum(x_vals * x_vals)
    
    mean = sum_x / (height * width)
    var = (sum_x2 / (height * width)) - (mean * mean)
    
    # Store statistics
    tl.store(mean_ptr + batch_idx * channels + channel_idx, mean)
    tl.store(var_ptr + batch_idx * channels + channel_idx, var)
    
    # Normalize and apply affine transformation
    for i in range(0, height * width, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < height * width
        x_vals = tl.load(x_ptr + channel_offset + offsets, mask=mask, other=0.0)
        
        # Normalize
        normalized = (x_vals - mean) / tl.sqrt(var + eps)
        
        # Apply affine transformation
        out_val = normalized * weight + bias
        
        # Store result
        tl.store(out_ptr + channel_offset + offsets, out_val, mask=mask)

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
        Applies optimized Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        batch_size, channels, height, width = x.shape
        
        # Ensure inputs are contiguous and on GPU
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Allocate memory for intermediate statistics
        mean = torch.empty(batch_size, channels, dtype=torch.float32, device='cuda')
        var = torch.empty(batch_size, channels, dtype=torch.float32, device='cuda')
        
        # Configure kernel launch parameters
        BLOCK_SIZE = 1024
        grid = (batch_size, channels)
        
        # Launch the fused kernel
        fused_instance_norm_matmul_kernel[grid](
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
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out