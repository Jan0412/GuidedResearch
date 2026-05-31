import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    num_features,  # Number of features
    total_elements,  # Total number of elements in the tensor
    eps,  # Epsilon value for numerical stability
    BLOCK_SIZE: tl.constexpr,
    FEATURE_BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of batches and other dimensions (excluding features)
    batch_idx = tl.program_id(0)
    
    # Calculate the offset to the start of this batch's data
    feature_stride = total_elements // num_features
    start_offset = batch_idx * feature_stride
    
    # Loop over features in blocks
    for feature_start in range(0, num_features, FEATURE_BLOCK_SIZE):
        feature_offsets = feature_start + tl.arange(0, FEATURE_BLOCK_SIZE)
        mask = feature_offsets < num_features
        
        # Compute x^2 for this feature block
        x_offsets = start_offset + feature_offsets * feature_stride
        x_block = tl.load(x_ptr + x_offsets, mask=mask)
        x_sq_block = x_block * x_block
        
        # Accumulate sum of squares across features
        # We'll compute the mean across the feature dimension
        # For now, just store the squared values
        # We'll do the reduction in a separate kernel or handle it differently
        
        # Actually, let's do a more efficient approach:
        # We'll compute the RMS in two passes or use a more efficient algorithm
        pass


# Better approach: Use two kernels - one to compute sum of squares per position, then normalize
@triton.jit
def compute_rms_kernel(
    x_ptr,  # Pointer to input tensor
    rms_sq_ptr,  # Pointer to output: mean of x^2 for each position (excluding feature dim)
    batch_size,  # Batch size
    num_features,  # Number of features
    spatial_size,  # Product of all dimensions except batch and feature
    eps,  # Epsilon value
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a spatial position (across batches)
    pos_idx = tl.program_id(0)
    
    # Compute the mean of x^2 across features for this spatial position
    sum_sq = 0.0
    for f in range(num_features):
        # Calculate the index in the flattened tensor
        # Layout: [batch, feature, spatial...] -> flat index = batch * (feature* spatial) + feature * spatial + pos_idx
        idx = pos_idx + f * spatial_size + pos_idx // spatial_size * num_features * spatial_size
        # Actually, better to compute the index directly
        batch_idx = pos_idx // spatial_size
        spatial_pos = pos_idx % spatial_size
        flat_idx = batch_idx * (num_features * spatial_size) + f * spatial_size + spatial_pos
        
        x_val = tl.load(x_ptr + flat_idx)
        sum_sq += x_val * x_val
    
    # Compute mean and add epsilon
    mean_sq = sum_sq / num_features
    rms_sq = mean_sq + eps
    tl.store(rms_sq_ptr + pos_idx, rms_sq)


@triton.jit
def normalize_kernel(
    x_ptr,  # Pointer to input tensor
    rms_sq_ptr,  # Pointer to precomputed (mean of x^2 + eps) values
    out_ptr,  # Pointer to output tensor
    batch_size,  # Batch size
    num_features,  # Number of features
    spatial_size,  # Product of all dimensions except batch and feature
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a spatial position
    pos_idx = tl.program_id(0)
    
    # Get the precomputed RMS squared value for this position
    rms_sq_val = tl.load(rms_sq_ptr + pos_idx)
    inv_rms = 1.0 / tl.sqrt(rms_sq_val)
    
    # Normalize all features for this spatial position
    batch_idx = pos_idx // spatial_size
    spatial_pos = pos_idx % spatial_size
    
    for f in range(num_features):
        flat_idx = batch_idx * (num_features * spatial_size) + f * spatial_size + spatial_pos
        x_val = tl.load(x_ptr + flat_idx)
        out_val = x_val * inv_rms
        tl.store(out_ptr + flat_idx, out_val)


def triton_rmsnorm(x: torch.Tensor, eps: float):
    """
    Apply RMS Normalization using Triton kernels.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, *)
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized tensor of same shape as input
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    batch_size = x.shape[0]
    num_features = x.shape[1]
    spatial_size = x.numel() // (batch_size * num_features)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Compute mean of x^2 + eps for each spatial position across batches and features
    # Total number of positions = batch_size * spatial_size
    total_positions = batch_size * spatial_size
    
    # Use reasonable block sizes
    BLOCK_SIZE = 256
    FEATURE_BLOCK_SIZE = 32  # For the compute kernel
    
    # Launch compute kernel
    rms_sq = torch.empty(total_positions, device=x.device, dtype=x.dtype)
    
    # Grid for compute kernel: one block per spatial position
    compute_grid = lambda meta: (total_positions,)
    compute_rms_kernel[compute_grid](
        x, rms_sq, batch_size, num_features, spatial_size, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Launch normalize kernel
    normalize_grid = lambda meta: (total_positions,)
    normalize_kernel[normalize_grid](
        x, rms_sq, out, batch_size, num_features, spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using custom Triton kernels.
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
        Applies RMS Normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_rmsnorm(x, self.eps)