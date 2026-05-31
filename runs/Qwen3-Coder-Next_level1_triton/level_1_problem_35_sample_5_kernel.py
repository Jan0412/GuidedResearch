import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    X,  # pointer to input tensor
    Y,  # pointer to output tensor
    Weight,  # pointer to gamma (scale)
    Bias,  # pointer to beta (shift)
    Mean,  # pointer to computed means (for debugging or reuse)
    Rstd,  # pointer to computed stds (for debugging or reuse)
    N,  # number of elements per group
    C,  # total number of channels
    G,  # number of groups
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one group (channel group)
    group_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    # Calculate offset for this batch and group
    # For input of shape (B, C, H, W), we treat as (B, G, C//G, H*W)
    channel_start = group_idx * (C // G)
    base_offset = batch_idx * C * N + channel_start * N
    
    # Compute mean and variance for this group
    # We'll compute the mean and variance in a single pass for better efficiency
    
    # Accumulators for mean and variance
    sum_x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate over elements in the group
    for i in range(0, N, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE) + i
        mask = offsets < N
        
        # Load data
        x_ptrs = X + base_offset + offsets
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        # Accumulate for mean/variance
        sum_x += x
        sum_x2 += x * x
    
    # Reduce within block
    # First sum across the block dimension
    sum_x = tl.sum(sum_x, axis=0)
    sum_x2 = tl.sum(sum_x2, axis=0)
    
    # We need a full reduction across blocks (handled by single block for simplicity)
    # For simplicity in this kernel, we assume BLOCK_SIZE >= N (or use multiple passes)
    # In practice, we'll set BLOCK_SIZE large enough to cover the entire group
    
    # Compute mean
    mean = sum_x / N
    
    # Compute variance
    var = sum_x2 / N - mean * mean
    
    # Compute standard deviation with epsilon for numerical stability
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and rstd if pointers are provided
    if Mean is not None:
        tl.store(Mean + batch_idx * G + group_idx, mean)
    if Rstd is not None:
        tl.store(Rstd + batch_idx * G + group_idx, rstd)
    
    # Now apply normalization and affine transformation
    # Load weight and bias for this group
    weight_val = tl.load(Weight + channel_start + group_idx).to(tl.float32) if Weight is not None else 1.0
    bias_val = tl.load(Bias + channel_start + group_idx).to(tl.float32) if Bias is not None else 0.0
    
    # Process elements again
    for i in range(0, N, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE) + i
        mask = offsets < N
        
        # Load input
        x_ptrs = X + base_offset + offsets
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize
        x_norm = (x - mean) * rstd
        
        # Apply affine transformation
        out = x_norm * weight_val + bias_val
        
        # Store output
        y_ptrs = Y + base_offset + offsets
        tl.store(y_ptrs + offsets, out.to(X.dtype.element_ty), mask=mask)


def triton_group_norm(x: torch.Tensor, num_groups: int, weight: torch.Tensor = None, bias: torch.Tensor = None, eps: float = 1e-5):
    """
    Apply Group Normalization using Triton kernel.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        num_groups: Number of groups (G)
        weight: Gamma parameter of shape (C,), optional
        bias: Beta parameter of shape (C,), optional
        eps: Small constant for numerical stability
    
    Returns:
        Output tensor of same shape as x
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get dimensions
    B, C, H, W = x.shape
    G = num_groups
    
    # Check if number of channels is divisible by groups
    assert C % G == 0, f"Number of channels ({C}) must be divisible by number of groups ({G})"
    
    # Per-group size
    N = C // G
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Prepare weight and bias if not provided
    if weight is None:
        weight = torch.ones(C, dtype=x.dtype, device=x.device)
    else:
        weight = weight.contiguous()
        
    if bias is None:
        bias = torch.zeros(C, dtype=x.dtype, device=x.device)
    else:
        bias = bias.contiguous()
    
    # Set block size - use a reasonable size for GPU
    # For better performance, we can use a larger block size since each program handles one group
    BLOCK_SIZE = 256  # Tunable parameter
    
    # Grid: one block per group, one block per batch
    grid = lambda meta: (G, B)
    
    # Launch kernel
    group_norm_kernel[grid](
        x, out, weight, bias, None, None,
        N, C, G, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization with custom Triton kernel.
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
        # Initialize learnable parameters (gamma and beta)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return triton_group_norm(x, self.num_groups, self.weight, self.bias)