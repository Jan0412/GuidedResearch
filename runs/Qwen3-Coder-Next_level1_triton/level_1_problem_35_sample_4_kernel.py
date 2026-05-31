import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    X,  # pointer to input
    Y,  # pointer to output
    Weight,  # pointer to gamma
    Bias,  # pointer to beta
    Mean,  # pointer to mean (output)
    Rstd,  # pointer to reciprocal std (output)
    N,  # total number of elements in X
    C,  # number of channels
    G,  # number of groups
    S,  # size of spatial dimensions (H*W*D etc)
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one (batch, group) pair
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    
    # Calculate the starting offset for this (batch, group) pair
    # Input shape: (batch_size, C, spatial_dims...)
    # Each group has C//G channels
    start_offset = batch_idx * (C * S) + group_idx * (C // G) * S
    
    # Compute statistics for this group
    sum = 0.0
    sum_sq = 0.0
    
    # Iterate over all elements in this group (C//G channels * S spatial elements)
    num_elements = (C // G) * S
    
    # Process in blocks for efficiency
    for start in range(0, num_elements, BLOCK_SIZE):
        off = start + tl.arange(0, BLOCK_SIZE)
        mask = off < num_elements
        
        # Load data
        data_ptr = X + start_offset + off
        data = tl.load(data_ptr, mask=mask, other=0.0)
        
        # Accumulate sums
        sum += tl.sum(data, axis=0)
        sum_sq += tl.sum(data * data, axis=0)
    
    # Compute mean and variance
    mean = sum / num_elements
    var = sum_sq / num_elements - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and rstd
    mean_ptr = Mean + batch_idx * G + group_idx
    rstd_ptr = Rstd + batch_idx * G + group_idx
    tl.store(mean_ptr, mean)
    tl.store(rstd_ptr, rstd)
    
    # Normalize and apply weight/bias
    # Iterate again to normalize and apply transformation
    weight_offset = group_idx * (C // G)
    bias_offset = group_idx * (C // G)
    
    for start in range(0, num_elements, BLOCK_SIZE):
        off = start + tl.arange(0, BLOCK_SIZE)
        mask = off < num_elements
        
        # Load data
        data_ptr = X + start_offset + off
        data = tl.load(data_ptr, mask=mask, other=0.0)
        
        # Normalize
        normalized = (data - mean) * rstd
        
        # Apply weight and bias
        # Weight and bias have shape (C,), so we need to map indices correctly
        channel_idx = (start + off) % (C // G)
        weight_idx = weight_offset + channel_idx
        bias_idx = bias_offset + channel_idx
        
        # Load weight and bias
        w = tl.load(Weight + weight_idx, mask=mask, other=0.0)
        b = tl.load(Bias + bias_idx, mask=mask, other=0.0)
        
        # Apply transformation
        result = normalized * w + b
        
        # Store result
        out_ptr = Y + start_offset + off
        tl.store(out_ptr, result, mask=mask)


def group_norm_triton(x, weight, bias, num_groups, eps):
    """
    Triton implementation of Group Normalization.
    
    Args:
        x: Input tensor of shape (batch_size, C, spatial_dims...)
        weight: Gamma tensor of shape (C,)
        bias: Beta tensor of shape (C,)
        num_groups: Number of groups
        eps: Epsilon for numerical stability
    
    Returns:
        Output tensor of same shape as x
    """
    batch_size, C, *spatial_dims = x.shape
    S = 1
    for d in spatial_dims:
        S *= d
        
    # Check input validity
    assert C % num_groups == 0, f"num_channels={C} must be divisible by num_groups={num_groups}"
    
    # Allocate output tensors
    y = torch.empty_like(x)
    mean = torch.empty(batch_size, num_groups, device=x.device, dtype=x.dtype)
    rstd = torch.empty(batch_size, num_groups, device=x.device, dtype=x.dtype)
    
    # Grid: one block per (batch, group) pair
    grid = (batch_size, num_groups)
    
    # Determine block size
    BLOCK_SIZE = 256
    
    # Launch kernel
    group_norm_kernel[grid](
        x, y, weight, bias, mean, rstd,
        x.numel(), C, num_groups, S, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized Group Normalization using Triton kernels.
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
        # Initialize weight and bias as in torch.nn.GroupNorm
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Call the Triton implementation
        return group_norm_triton(x, self.weight, self.bias, self.num_groups, self.eps)