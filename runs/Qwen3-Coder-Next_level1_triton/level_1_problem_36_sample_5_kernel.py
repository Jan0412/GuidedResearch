import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    weight_ptr,  # Pointer to scale parameter (gamma)
    n_elements,  # Total number of elements in input/output
    num_features,  # Number of features (dimension 1)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature map across all batch and other dimensions
    # We'll process along dimension 1 (features)
    
    # Program ID corresponds to feature index
    feat_id = tl.program_id(0)
    
    # Calculate offsets for this feature
    # We'll iterate over all other dimensions for this feature
    # Total elements per feature = n_elements / num_features
    elements_per_feature = n_elements // num_features
    
    # For each feature, we process elements in blocks
    for start_idx in range(0, elements_per_feature, BLOCK_SIZE):
        offsets = feat_id * elements_per_feature + start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load the values for this feature at these positions
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute squared values
        x_sq = x * x
        
        # We'll accumulate the sum of squares across all positions for this feature
        # But since we're doing it in a loop, we need to accumulate in a register
        # For simplicity, we'll do a two-pass approach: first compute sum of squares, then normalize
        
        # For now, store x for later processing
        tl.store(out_ptr + offsets, x, mask=mask)


# For simplicity and performance, we'll use a two-kernel approach:
# 1. Compute sum of squares per feature
# 2. Normalize using those sums

@triton.jit
def compute_rms_kernel(
    x_ptr,  # Pointer to input tensor
    rms_ptr,  # Pointer to output RMS values (one per feature)
    batch_dim,  # Product of dimensions except feature dimension
    num_features,  # Number of features
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature
    feat_id = tl.program_id(0)
    
    # Compute sum of squares for this feature across all other dimensions
    sum_sq = 0.0
    for i in range(0, batch_dim, BLOCK_SIZE):
        offsets = feat_id * batch_dim + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < batch_dim * feat_id + batch_dim
        
        # Load values for this feature
        # Need to adjust offset calculation - for each feature, we have batch_dim elements
        offsets = feat_id * batch_dim + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (feat_id + 1) * batch_dim
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    
    # Compute mean and then RMS
    mean_sq = sum_sq / batch_dim
    rms = tl.sqrt(mean_sq + eps)
    
    # Store RMS for this feature
    tl.store(rms_ptr + feat_id, rms)


@triton.jit
def normalize_kernel(
    x_ptr,  # Pointer to input tensor
    rms_ptr,  # Pointer to RMS values
    out_ptr,  # Pointer to output tensor
    batch_dim,  # Product of dimensions except feature dimension
    num_features,  # Number of features
    BLOCK_SIZE: tl.constexpr,
):
    # Process in blocks along the non-feature dimensions
    batch_block_id = tl.program_id(0)
    
    for feat_id in range(num_features):
        # Compute offsets for this feature and batch block
        start_idx = batch_block_id * BLOCK_SIZE + feat_id * batch_dim
        end_idx = min(start_idx + BLOCK_SIZE, (feat_id + 1) * batch_dim)
        
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        
        # Load input values
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Load RMS for this feature
        rms = tl.load(rms_ptr + feat_id)
        
        # Normalize
        out = x / rms
        
        # Store result
        tl.store(out_ptr + offsets, out, mask=mask)


# Better approach: single kernel that computes RMS on-the-fly
@triton.jit
def fused_rms_norm_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    weight_ptr,  # Pointer to scale parameter (gamma) - unused but kept for compatibility
    n_elements,  # Total number of elements in input/output
    batch_dim,  # Product of dimensions except feature dimension
    num_features,  # Number of features
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature
    feat_id = tl.program_id(0)
    
    # First pass: compute sum of squares for this feature
    sum_sq = 0.0
    for i in range(0, batch_dim, BLOCK_SIZE):
        offsets = feat_id * batch_dim + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (feat_id + 1) * batch_dim
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)
    
    # Compute RMS
    mean_sq = sum_sq / batch_dim
    rms = tl.sqrt(mean_sq + eps)
    
    # Second pass: normalize
    for i in range(0, batch_dim, BLOCK_SIZE):
        offsets = feat_id * batch_dim + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (feat_id + 1) * batch_dim
        
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = x / rms
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float = 1e-5):
    """
    Applies RMS Normalization using Triton kernels.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *)
        eps (float): Small value to avoid division by zero
    
    Returns:
        torch.Tensor: Normalized output tensor
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get dimensions
    batch_size = x.size(0)
    num_features = x.size(1)
    other_dims = x.size()[2:]
    batch_dim = 1
    for d in other_dims:
        batch_dim *= d
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size
    BLOCK_SIZE = 1024
    
    # Grid: one program per feature
    grid = (num_features,)
    
    # Launch kernel
    fused_rms_norm_kernel[grid](
        x, out, None,  # weight_ptr (unused)
        x.numel(),
        batch_dim,
        num_features,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using Triton kernels.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_rms_norm(x, self.eps)