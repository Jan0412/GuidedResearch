import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def group_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    num_features,
    height,
    width,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global thread index
    pid = tl.program_id(0)
    # Each block handles one feature channel
    feature_idx = pid
    
    if feature_idx >= num_features:
        return
        
    # Calculate group information
    channels_per_group = num_features // num_groups
    group_idx = feature_idx // channels_per_group
    
    # Shared memory for reduction
    mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Initialize accumulators
    sum_val = 0.0
    sum_sq = 0.0
    
    # Process all elements for this feature channel
    for i in range(0, batch_size * height * width, BLOCK_SIZE):
        # Calculate actual indices
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < batch_size * height * width
        
        # Load data
        x_data = tl.load(x_ptr + idx * num_features + feature_idx, mask=mask, other=0.0)
        
        # Accumulate sum and sum of squares
        sum_val += tl.sum(x_data)
        sum_sq += tl.sum(x_data * x_data)
    
    # Reduce across threads in block
    sum_val = tl.sum(sum_val)
    sum_sq = tl.sum(sum_sq)
    
    # Compute mean and variance
    total_elements = batch_size * height * width
    mean_val = sum_val / total_elements
    var_val = sum_sq / total_elements - mean_val * mean_val
    
    # Compute reciprocal standard deviation
    rstd_val = 1.0 / tl.sqrt(var_val + eps)
    
    # Store mean and rstd for this feature channel
    tl.store(mean_ptr + feature_idx, mean_val)
    tl.store(rstd_ptr + feature_idx, rstd_val)
    
    # Normalize and apply affine transformation
    for i in range(0, batch_size * height * width, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < batch_size * height * width
        
        # Load input
        x_data = tl.load(x_ptr + idx * num_features + feature_idx, mask=mask, other=0.0)
        
        # Normalize
        normalized = (x_data - mean_val) * rstd_val
        
        # Apply weight and bias
        weight_val = tl.load(weight_ptr + feature_idx, mask=True, other=1.0)
        bias_val = tl.load(bias_ptr + feature_idx, mask=True, other=0.0)
        
        out_val = normalized * weight_val + bias_val
        
        # Store output
        tl.store(out_ptr + idx * num_features + feature_idx, out_val, mask=mask)

# More efficient implementation using proper reduction and shared memory
@triton.jit
def group_norm_kernel_optimized(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    num_features,
    height,
    width,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread and block information
    tid = tl.thread_id()
    bid = tl.program_id(0)
    
    # Each block processes one feature channel
    feature_idx = bid
    
    if feature_idx >= num_features:
        return
        
    # Calculate group info
    channels_per_group = num_features // num_groups
    group_idx = feature_idx // channels_per_group
    
    # Shared memory for reduction
    shared_mean = tl.shared_memory(shape=(BLOCK_SIZE,), dtype=tl.float32)
    shared_var = tl.shared_memory(shape=(BLOCK_SIZE,), dtype=tl.float32)
    
    # Initialize accumulators
    sum_val = 0.0
    sum_sq = 0.0
    
    # Process all elements for this feature channel
    total_elements = batch_size * height * width
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Reduction loop
    for i in range(num_blocks):
        offset = i * BLOCK_SIZE + tid
        if offset < total_elements:
            x_data = tl.load(x_ptr + offset * num_features + feature_idx)
            sum_val += x_data
            sum_sq += x_data * x_data
    
    # Synchronize threads within block
    tl.sync()
    
    # Store partial sums in shared memory
    shared_mean[tid] = sum_val
    shared_var[tid] = sum_sq
    tl.sync()
    
    # Final reduction
    if tid == 0:
        final_sum = 0.0
        final_sum_sq = 0.0
        for i in range(BLOCK_SIZE):
            final_sum += shared_mean[i]
            final_sum_sq += shared_var[i]
        
        # Compute mean and variance
        mean_val = final_sum / total_elements
        var_val = final_sum_sq / total_elements - mean_val * mean_val
        
        # Compute reciprocal standard deviation
        rstd_val = 1.0 / tl.sqrt(var_val + eps)
        
        # Store mean and rstd
        tl.store(mean_ptr + feature_idx, mean_val)
        tl.store(rstd_ptr + feature_idx, rstd_val)
    
    tl.sync()
    
    # Now compute output for all elements
    for i in range(0, total_elements, BLOCK_SIZE):
        offset = i + tid
        if offset < total_elements:
            # Load input
            x_data = tl.load(x_ptr + offset * num_features + feature_idx)
            
            # Load precomputed stats
            mean_val = tl.load(mean_ptr + feature_idx)
            rstd_val = tl.load(rstd_ptr + feature_idx)
            
            # Normalize
            normalized = (x_data - mean_val) * rstd_val
            
            # Apply weight and bias
            weight_val = tl.load(weight_ptr + feature_idx)
            bias_val = tl.load(bias_ptr + feature_idx)
            
            out_val = normalized * weight_val + bias_val
            
            # Store output
            tl.store(out_ptr + offset * num_features + feature_idx, out_val)

# Even better version with proper group handling
@triton.jit
def group_norm_kernel_fused(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    num_features,
    height,
    width,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature channel
    feature_idx = tl.program_id(0)
    
    if feature_idx >= num_features:
        return
        
    # Calculate group info
    channels_per_group = num_features // num_groups
    group_idx = feature_idx // channels_per_group
    
    # Compute mean and variance for this feature channel
    total_elements = batch_size * height * width
    
    # Initialize accumulators
    sum_val = 0.0
    sum_sq = 0.0
    
    # Loop through all elements for this channel
    for i in range(total_elements):
        x_data = tl.load(x_ptr + i * num_features + feature_idx)
        sum_val += x_data
        sum_sq += x_data * x_data
    
    # Reduce across all elements for this channel
    sum_val = tl.sum(sum_val)
    sum_sq = tl.sum(sum_sq)
    
    # Compute mean and variance
    mean_val = sum_val / total_elements
    var_val = sum_sq / total_elements - mean_val * mean_val
    
    # Compute reciprocal standard deviation
    rstd_val = 1.0 / tl.sqrt(var_val + eps)
    
    # Store mean and rstd
    tl.store(mean_ptr + feature_idx, mean_val)
    tl.store(rstd_ptr + feature_idx, rstd_val)
    
    # Apply normalization and affine transformation
    for i in range(total_elements):
        x_data = tl.load(x_ptr + i * num_features + feature_idx)
        
        # Normalize
        normalized = (x_data - mean_val) * rstd_val
        
        # Apply weight and bias
        weight_val = tl.load(weight_ptr + feature_idx)
        bias_val = tl.load(bias_ptr + feature_idx)
        
        out_val = normalized * weight_val + bias_val
        
        # Store output
        tl.store(out_ptr + i * num_features + feature_idx, out_val)

def triton_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                     num_groups: int, eps: float = 1e-5):
    """
    Triton-based Group Normalization implementation
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda and bias.is_cuda, "Weight and bias must be on CUDA"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    batch_size, num_features, height, width = x.shape
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Allocate intermediate tensors for mean and rstd
    mean = torch.empty(num_features, dtype=torch.float32, device=x.device)
    rstd = torch.empty(num_features, dtype=torch.float32, device=x.device)
    
    # Determine grid size
    grid_size = num_features
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    group_norm_kernel_fused[grid_size](
        x, weight, bias, out, mean, rstd,
        batch_size, num_features, height, width, num_groups, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized Model with Triton-based Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = 1e-5
        
        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Call our Triton-based group norm
        return triton_group_norm(x, self.weight, self.bias, self.num_groups, self.eps)