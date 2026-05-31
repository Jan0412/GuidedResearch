import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def instance_norm_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
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
    shared_var = tl.shared_ptr(var_ptr, BLOCK_SIZE)
    
    # Calculate offset for this batch and channel
    batch_offset = batch_idx * channels * height * width
    channel_offset = channel_idx * height * width
    
    # Load data for this thread
    offsets = batch_offset + channel_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_idx + 1) * channels * height * width
    
    # Load input data
    x_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean for this channel across batch and spatial dimensions
    mean = tl.sum(x_data) / (height * width)
    tl.store(shared_mean + tl.program_id(2), mean)
    
    # Compute variance
    diff = x_data - mean
    var = tl.sum(diff * diff) / (height * width)
    tl.store(shared_var + tl.program_id(2), var)
    
    # Synchronize threads
    tl.sync()
    
    # Compute final normalization
    if tl.program_id(2) == 0:
        # All threads compute the final normalized value
        mean_val = tl.load(mean_ptr + channel_idx)
        var_val = tl.load(var_ptr + channel_idx)
        std_val = tl.sqrt(var_val + eps)
        
        # Load weight and bias
        weight = tl.load(weight_ptr + channel_idx)
        bias = tl.load(bias_ptr + channel_idx)
        
        # Apply normalization
        for i in range(BLOCK_SIZE):
            if offsets[i] < (batch_idx + 1) * channels * height * width:
                normalized = (x_data[i] - mean_val) / std_val
                out_val = normalized * weight + bias
                tl.store(out_ptr + offsets[i], out_val)

@triton.jit
def instance_norm_forward_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
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
    batch_offset = batch_idx * channels * height * width
    channel_offset = channel_idx * height * width
    
    # Process in chunks
    for chunk_start in range(0, height * width, BLOCK_SIZE):
        # Calculate actual offsets
        chunk_end = min(chunk_start + BLOCK_SIZE, height * width)
        chunk_size = chunk_end - chunk_start
        
        # Load data for this chunk
        offsets = batch_offset + channel_offset + tl.arange(chunk_start, chunk_end)
        mask = offsets < (batch_idx + 1) * channels * height * width
        
        # Load input data
        x_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute mean and variance for this chunk
        mean = tl.sum(x_data) / chunk_size
        diff = x_data - mean
        var = tl.sum(diff * diff) / chunk_size
        
        # Normalize
        std_val = tl.sqrt(var + eps)
        
        # Load weight and bias
        weight = tl.load(weight_ptr + channel_idx)
        bias = tl.load(bias_ptr + channel_idx)
        
        # Apply normalization
        normalized = (x_data - mean) / std_val
        out_val = normalized * weight + bias
        
        # Store result
        tl.store(out_ptr + offsets, out_val, mask=mask)

# Optimized version using more efficient reduction patterns
@triton.jit
def fused_instance_norm_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    batch_size,
    channels,
    height,
    width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate offsets
    batch_offset = batch_idx * channels * height * width
    channel_offset = channel_idx * height * width
    
    # Initialize accumulators
    sum_x = 0.0
    sum_x2 = 0.0
    
    # Process all elements for this channel
    for i in range(height * width):
        offset = batch_offset + channel_offset + i
        x = tl.load(x_ptr + offset)
        sum_x += x
        sum_x2 += x * x
    
    # Compute mean and variance
    total_elements = height * width
    mean = sum_x / total_elements
    var = sum_x2 / total_elements - mean * mean
    
    # Compute standard deviation
    std = tl.sqrt(var + eps)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Apply normalization and store results
    for i in range(height * width):
        offset = batch_offset + channel_offset + i
        x = tl.load(x_ptr + offset)
        normalized = (x - mean) / std
        out_val = normalized * weight + bias
        tl.store(out_ptr + offset, out_val)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Use custom Triton kernel for instance normalization
        batch_size, channels, height, width = x.shape
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Define block size
        BLOCK_SIZE = 1024
        
        # Calculate grid dimensions
        grid = (
            batch_size, 
            channels, 
            1
        )
        
        # Launch kernel
        fused_instance_norm_kernel[grid](
            x,
            out,
            self.weight,
            self.bias,
            batch_size,
            channels,
            height,
            width,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out