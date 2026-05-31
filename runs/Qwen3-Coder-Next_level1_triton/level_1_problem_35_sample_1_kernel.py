import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    X,  # pointer to input tensor
    Y,  # pointer to output tensor
    Weight,  # pointer to gamma (scale) parameter
    Bias,  # pointer to beta (shift) parameter
    Mean,  # pointer to mean tensor (output of forward pass, not used in inference but kept for consistency)
    Var,  # pointer to variance tensor (output of forward pass)
    N,  # total number of elements in the input tensor
    C,  # number of channels
    G,  # number of groups
    D,  # spatial dimensions per group (C//G * spatial_size)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one group for one sample in the batch
    # Total programs = batch_size * G
    batch_idx = tl.program_id(0) // G
    group_idx = tl.program_id(0) % G

    # Calculate start indices
    # Each group processes C//G channels, and each channel has D spatial elements
    # So each program handles (C//G) * D = D elements
    start_idx = (batch_idx * C + group_idx * (C // G)) * D

    # Compute mean and variance in a single pass (online algorithm for numerical stability)
    # We'll accumulate sum and sum of squares
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Process the elements in blocks
    for offset in range(0, D, BLOCK_SIZE):
        offsets = start_idx + offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < start_idx + D
        x_val = tl.load(X + offsets, mask=mask).to(tl.float32)
        sum_val += tl.where(mask, x_val, 0.0)
        sum_sq_val += tl.where(mask, x_val * x_val, 0.0)

    # Reduce across BLOCK_SIZE dimension
    block_sum = tl.sum(sum_val, axis=0)
    block_sum_sq = tl.sum(sum_sq_val, axis=0)

    # Since each program handles D elements, we need to compute mean and variance
    # We use a simple reduction (could be optimized with more advanced techniques for large D)
    mean = block_sum / D
    var = block_sum_sq / D - mean * mean

    # Normalize
    std = tl.sqrt(var + eps)
    inv_std = 1.0 / std

    # Now apply weight and bias
    # We'll write out normalized values in a separate pass to keep it simple and correct
    for offset in range(0, D, BLOCK_SIZE):
        offsets = start_idx + offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < start_idx + D
        x_val = tl.load(X + offsets, mask=mask).to(tl.float32)
        # Normalize: (x - mean) / std
        x_norm = (x_val - mean) * inv_std
        # Apply weight and bias
        # Weight and Bias are per-channel, so we need to adjust for the group offset
        channel_idx = group_idx * (C // G) + (offsets - start_idx) // D
        w_val = tl.load(Weight + channel_idx, mask=mask)
        b_val = tl.load(Bias + channel_idx, mask=mask)
        y_val = x_norm * w_val + b_val
        tl.store(Y + offsets, y_val.to(Y.dtype.element_ty), mask=mask)


def triton_group_norm(x, weight, bias, num_groups, eps=1e-5):
    """
    Triton implementation of Group Normalization.
    
    Args:
        x: Input tensor of shape (batch_size, num_channels, *spatial_dims)
        weight: Gamma (scale) parameter of shape (num_channels,)
        bias: Beta (shift) parameter of shape (num_channels,)
        num_groups: Number of groups (G)
        eps: Small constant for numerical stability
    
    Returns:
        Output tensor of same shape as x
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    batch_size, num_channels = x.shape[0], x.shape[1]
    spatial_size = x.numel() // (batch_size * num_channels)
    num_groups = min(num_groups, num_channels)
    assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"
    
    # Calculate dimensions
    channels_per_group = num_channels // num_groups
    elements_per_group = channels_per_group * spatial_size
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Grid: batch_size * num_groups (one program per group per batch)
    grid = (batch_size * num_groups,)
    
    # Block size: tune based on spatial size
    BLOCK_SIZE = min(128, elements_per_group)
    
    # Launch kernel
    group_norm_kernel[grid](
        x, out, weight, bias, None, None,
        x.numel(),
        num_channels,
        num_groups,
        elements_per_group,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using Triton kernel.
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
        # Initialize learnable parameters
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
        return triton_group_norm(x, self.weight, self.bias, self.num_groups, self.eps)