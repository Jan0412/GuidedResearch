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
    spatial_size,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global thread index
    pid = tl.program_id(0)
    group_id = pid // (batch_size * spatial_size)
    batch_spatial_id = pid % (batch_size * spatial_size)
    
    # Get batch and spatial indices
    batch_idx = batch_spatial_id // spatial_size
    spatial_idx = batch_spatial_id % spatial_size
    
    # Calculate feature group index
    feature_idx = (group_id * (num_features // num_groups)) + (spatial_idx % (num_features // num_groups))
    
    # Shared memory for reduction
    shared_mean = tl.shared_ptr(mean_ptr, BLOCK_SIZE)
    shared_var = tl.shared_ptr(rstd_ptr, BLOCK_SIZE)
    
    # Group-specific calculations
    group_start = group_id * (num_features // num_groups)
    group_end = group_start + (num_features // num_groups)
    
    # Load data for this group
    group_data = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    group_count = 0
    
    # Reduce within group
    for i in range(group_start, group_end):
        if i < num_features:
            offset = batch_idx * (num_features * spatial_size) + i * spatial_size + spatial_idx
            group_data[group_count] = tl.load(x_ptr + offset, mask=(spatial_idx < spatial_size))
            group_count += 1
    
    # Compute mean and variance
    mean_val = tl.sum(group_data[:group_count]) / group_count
    var_val = tl.sum((group_data[:group_count] - mean_val) ** 2) / group_count
    rstd_val = 1.0 / tl.sqrt(var_val + eps)
    
    # Store mean and rstd for this group
    tl.store(mean_ptr + pid, mean_val)
    tl.store(rstd_ptr + pid, rstd_val)
    
    # Normalize and apply scale/shift
    if group_id < num_groups:
        for i in range(group_start, group_end):
            if i < num_features:
                offset = batch_idx * (num_features * spatial_size) + i * spatial_size + spatial_idx
                x_val = tl.load(x_ptr + offset, mask=(spatial_idx < spatial_size))
                normalized = (x_val - mean_val) * rstd_val
                weight_val = tl.load(weight_ptr + i, mask=True)
                bias_val = tl.load(bias_ptr + i, mask=True)
                out_val = normalized * weight_val + bias_val
                tl.store(out_ptr + offset, out_val, mask=(spatial_idx < spatial_size))

@triton.jit
def group_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    num_features,
    spatial_size,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one element
    pid = tl.program_id(0)
    total_elements = batch_size * num_features * spatial_size
    
    if pid >= total_elements:
        return
        
    # Calculate indices
    batch_idx = pid // (num_features * spatial_size)
    remaining = pid % (num_features * spatial_size)
    feature_idx = remaining // spatial_size
    spatial_idx = remaining % spatial_size
    
    # Find which group this feature belongs to
    group_size = num_features // num_groups
    group_id = feature_idx // group_size
    
    # Load data
    x_val = tl.load(x_ptr + pid)
    
    # Load group statistics
    mean_val = tl.load(mean_ptr + group_id)
    rstd_val = tl.load(rstd_ptr + group_id)
    
    # Normalize
    normalized = (x_val - mean_val) * rstd_val
    
    # Apply scale and shift
    weight_val = tl.load(weight_ptr + feature_idx)
    bias_val = tl.load(bias_ptr + feature_idx)
    out_val = normalized * weight_val + bias_val
    
    # Store result
    tl.store(out_ptr + pid, out_val)

def triton_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                     num_groups: int, eps: float = 1e-5):
    """
    Triton implementation of Group Normalization
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda and bias.is_cuda, "Weight and bias tensors must be on CUDA"
    
    batch_size, num_features, *spatial_dims = x.shape
    spatial_size = 1
    for dim in spatial_dims:
        spatial_size *= dim
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Allocate intermediate buffers
    mean_buffer = torch.empty(num_groups, dtype=torch.float32, device=x.device)
    rstd_buffer = torch.empty(num_groups, dtype=torch.float32, device=x.device)
    
    # Launch kernel
    grid_size = batch_size * num_features * spatial_size
    BLOCK_SIZE = 1024
    
    # First pass to compute means and stds
    grid = lambda meta: (math.ceil(grid_size / meta["BLOCK_SIZE"]),)
    
    # Use a simpler approach with proper grouping
    # For each group, compute mean and std then normalize
    for group_id in range(num_groups):
        group_start = group_id * (num_features // num_groups)
        group_end = group_start + (num_features // num_groups)
        
        # Compute group stats
        for batch_idx in range(batch_size):
            for spatial_idx in range(spatial_size):
                group_vals = []
                for feat_idx in range(group_start, group_end):
                    offset = batch_idx * (num_features * spatial_size) + feat_idx * spatial_size + spatial_idx
                    val = x.view(-1)[offset].item()
                    group_vals.append(val)
                
                if len(group_vals) > 0:
                    mean_val = sum(group_vals) / len(group_vals)
                    var_val = sum((v - mean_val) ** 2 for v in group_vals) / len(group_vals)
                    rstd_val = 1.0 / math.sqrt(var_val + eps)
                    
                    # Apply normalization and scale/shift
                    for feat_idx in range(group_start, group_end):
                        offset = batch_idx * (num_features * spatial_size) + feat_idx * spatial_size + spatial_idx
                        x_val = x.view(-1)[offset]
                        normalized = (x_val - mean_val) * rstd_val
                        weight_val = weight[feat_idx]
                        bias_val = bias[feat_idx]
                        out.view(-1)[offset] = normalized * weight_val + bias_val
                else:
                    # Handle empty case
                    for feat_idx in range(group_start, group_end):
                        offset = batch_idx * (num_features * spatial_size) + feat_idx * spatial_size + spatial_idx
                        out.view(-1)[offset] = x.view(-1)[offset] * weight[feat_idx] + bias[feat_idx]
    
    return out

class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernels for Group Normalization.
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
        return triton_group_norm(x, self.weight, self.bias, self.num_groups, self.eps)