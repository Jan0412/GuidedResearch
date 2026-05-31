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
    channels,
    height,
    width,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global thread index
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total elements
    total_elements = batch_size * channels * height * width
    
    # Mask to ensure we don't go out of bounds
    mask = idx < total_elements
    
    # Reshape linear index to 4D coordinates
    # We'll process elements in batches of channels grouped by num_groups
    batch_idx = idx // (channels * height * width)
    remaining = idx % (channels * height * width)
    channel_idx = remaining // (height * width)
    remaining = remaining % (height * width)
    h_idx = remaining // width
    w_idx = remaining % width
    
    # Calculate which group this channel belongs to
    group_idx = channel_idx // (channels // num_groups)
    
    # Shared memory for reduction operations
    shared_mean = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    shared_var = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Load data
    x_vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
    
    # Compute group statistics
    # For each group, compute mean and variance
    group_start = batch_idx * channels * height * width + group_idx * (channels // num_groups) * height * width
    group_end = group_start + (channels // num_groups) * height * width
    
    # Compute mean for the group
    group_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    group_count = 0
    
    # Process elements in group
    for i in range(group_start, group_end, BLOCK_SIZE):
        group_idx_local = i + tl.arange(0, BLOCK_SIZE)
        group_mask = (group_idx_local < group_end) & (group_idx_local < total_elements)
        group_vals = tl.load(x_ptr + group_idx_local, mask=group_mask, other=0.0)
        group_sum += group_vals
        group_count += tl.sum(tl.where(group_mask, 1, 0))
    
    # Reduce to get mean
    mean_val = tl.sum(group_sum) / group_count
    
    # Compute variance
    var_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(group_start, group_end, BLOCK_SIZE):
        group_idx_local = i + tl.arange(0, BLOCK_SIZE)
        group_mask = (group_idx_local < group_end) & (group_idx_local < total_elements)
        group_vals = tl.load(x_ptr + group_idx_local, mask=group_mask, other=0.0)
        diff = group_vals - mean_val
        var_sum += diff * diff
    
    var_val = tl.sum(var_sum) / group_count
    
    # Add epsilon for numerical stability
    rstd_val = 1.0 / tl.sqrt(var_val + eps)
    
    # Normalize and apply affine transformation
    normalized = (x_vals - mean_val) * rstd_val
    weight_val = tl.load(weight_ptr + channel_idx, mask=channel_idx < channels, other=0.0)
    bias_val = tl.load(bias_ptr + channel_idx, mask=channel_idx < channels, other=0.0)
    
    out_val = normalized * weight_val + bias_val
    
    # Store results
    tl.store(out_ptr + idx, out_val, mask=mask)
    
    # Store mean and rstd for this group
    if group_idx == 0 and batch_idx == 0:
        tl.store(mean_ptr + group_idx, mean_val)
        tl.store(rstd_ptr + group_idx, rstd_val)

def triton_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                     num_groups: int, eps: float = 1e-5):
    """
    Triton implementation of Group Normalization
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda and bias.is_cuda, "Weight and bias tensors must be on CUDA"
    
    batch_size, channels, height, width = x.shape
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Allocate memory for intermediate results
    means = torch.empty(num_groups, dtype=torch.float32, device=x.device)
    rstds = torch.empty(num_groups, dtype=torch.float32, device=x.device)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    group_norm_kernel[grid](
        x, weight, bias, out, means, rstds,
        batch_size, channels, height, width, num_groups, eps, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer with optimized Triton kernel.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return triton_group_norm(x, self.weight, self.bias, self.num_groups, self.eps)