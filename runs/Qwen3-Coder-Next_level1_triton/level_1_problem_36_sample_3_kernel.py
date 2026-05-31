import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    n_elements,  # Total number of elements
    batch_size: tl.constexpr,
    num_features: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature channel across all batches and spatial locations
    feature_idx = tl.program_id(0)
    
    # Calculate offsets for this feature channel
    # We process one feature at a time across all batches and spatial locations
    offsets = feature_idx + tl.arange(0, BLOCK_SIZE) * num_features
    
    # Initialize accumulator for RMS computation
    sum_sq = 0.0
    
    # Compute sum of squares for this feature channel
    for i in range(0, n_elements, num_features * BLOCK_SIZE):
        # Adjust offset for the current iteration
        curr_offsets = offsets + i
        mask = curr_offsets < n_elements
        x = tl.load(x_ptr + curr_offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    
    # Compute mean and RMS
    total_elements = batch_size * spatial_size
    mean_sq = sum_sq / total_elements
    rms = tl.sqrt(mean_sq + 1e-5)
    
    # Normalize and store output
    for i in range(0, n_elements, num_features * BLOCK_SIZE):
        curr_offsets = offsets + i
        mask = curr_offsets < n_elements
        x = tl.load(x_ptr + curr_offsets, mask=mask, other=0.0)
        out = x / rms
        tl.store(out_ptr + curr_offsets, out, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float = 1e-5):
    """
    Applies RMS Normalization using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, dim1, dim2)
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized tensor with same shape as input
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    batch_size, num_features, dim1, dim2 = x.shape
    spatial_size = dim1 * dim2
    n_elements = batch_size * num_features * spatial_size
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # For the kernel, we'll launch one block per feature channel
    # But since we want to process multiple features per block for better occupancy,
    # we'll use a grid that allows parallel processing
    
    # Use a reasonable block size - adjust based on spatial dimensions
    BLOCK_SIZE = 128
    
    # Grid: one block per feature to handle the reduction properly
    grid = (num_features,)
    
    # Launch kernel
    rms_norm_kernel[grid](
        x, out, n_elements,
        batch_size, num_features, spatial_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized RMS Normalization model using Triton kernel.
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
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_rms_norm(x, self.eps)