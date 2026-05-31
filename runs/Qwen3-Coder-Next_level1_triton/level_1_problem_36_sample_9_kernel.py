import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    num_features,  # Number of features (second dimension)
    total_elements,  # Total number of elements in the tensor
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate which batch/sample we're processing
    # Each block processes one feature dimension across all other dimensions
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Calculate the total number of elements per feature (all dimensions except batch and feature)
    num_per_feature = total_elements // (batch_idx + 1) // num_features if batch_idx == 0 else total_elements // batch_idx // num_features
    # Actually, better to compute stride-based indexing
    # For tensor shape (B, F, *), the stride for feature dim is prod(other_dims)
    # We'll use a simpler approach: calculate offsets directly
    
    # Get the starting offset for this (batch, feature) combination
    # We'll process feature dimension elements in blocks
    
    # Calculate offset for this batch and feature
    # For shape (B, F, D1, D2, ...), the stride for feature is D1*D2*...
    # But we'll use a more direct approach
    
    # Compute RMS for this feature across all other dimensions
    # Each feature is processed by multiple threads in parallel
    # We'll compute partial sums and then combine
    
    # For simplicity, let's process one feature dimension at a time
    # Each (batch, feature) combination is handled by one block
    # But for efficiency, we can process multiple features per block
    
    # Let's restructure: process one batch at a time, with each block handling multiple features
    # Actually, let's do one feature at a time per block for simplicity and good parallelism
    
    # Get the starting index for this feature in this batch
    # We'll compute the mean of squares for this feature across all non-feature dimensions
    
    # Calculate stride for feature dimension
    # Assuming input is (B, F, D1, D2), stride for feature dim is D1*D2
    # Let's compute this from total_elements and num_features
    # total_elements = B * F * D1 * D2
    # So elements per (B,F) = D1*D2 = total_elements // (B * F)
    
    # Actually, we can compute this more carefully:
    # total_elements = batch_size * num_features * prod(other_dims)
    # Let's pass in the product of other dimensions explicitly to avoid confusion
    
    # Since the kernel is complex with dynamic shapes, let's simplify:
    # We'll process one batch at a time, with each block handling one feature
    # Each thread processes a contiguous chunk of the feature data
    
    # For the current (batch, feature) combination, compute RMS
    # The feature data starts at: batch_idx * (num_features * other_dims) + feature_idx * other_dims
    # other_dims = total_elements // (batch_size * num_features)
    
    # But we don't have batch_size directly, so let's compute it differently
    
    # Alternative approach: iterate through all elements and accumulate for each (batch, feature)
    # This is inefficient for large tensors, but let's use a better method
    
    # Let's assume the input is always 4D: (B, F, D1, D2) for simplicity as per the example
    # This makes the kernel simpler and more efficient
    
    # Since the kernel parameters are limited, let's use a different approach
    # We'll process the tensor in a way that each block handles a slice of the feature dimension
    
    # Actually, let's rewrite the kernel to be simpler and more efficient
    # Process one batch at a time, with each block handling multiple features in parallel
    
    # Calculate total elements in the non-feature dimensions
    # other_dims = total_elements // batch_size // num_features
    
    # For now, let's use a simpler kernel that processes the entire tensor with proper indexing
    
    # Calculate the index for the current thread
    # Each block handles one feature across all batches and other dimensions
    
    # Let's do a 2D grid: [batch_idx, feature_idx]
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Calculate other_dims product
    # We'll assume 4D tensor as in the example
    # Actually, let's make it more general by computing strides
    
    # For tensor (B, F, D1, D2), stride for feature is D1*D2
    # Let's compute this as total_elements // (batch_size * num_features)
    # But we need batch_size, so let's pass it as a parameter
    
    # Since the kernel is getting complex, let's simplify the approach
    # We'll process the tensor in a row-major fashion, computing RMS for each (batch, feature) pair
    
    # Let's use a 1D grid and compute the (batch, feature) indices from the block index
    # But that's inefficient for large tensors
    
    # Final approach: use a 2D grid [batch_size, num_features], and each block processes
    # a slice of the feature data
    
    # Calculate the number of elements in the non-feature dimensions
    # We'll compute this as total_elements // batch_size // num_features
    # But we need batch_size, so let's pass it as a parameter
    
    # Let's modify the kernel to take batch_size explicitly
    pass  # This is a placeholder - we'll rewrite the kernel properly below


@triton.jit
def rms_norm_kernel_v2(
    x_ptr,
    out_ptr,
    batch_size,
    num_features,
    other_dims,  # Product of all dimensions except batch and feature
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: [batch_size, num_features]
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Calculate the starting offset for this (batch, feature) combination
    # Each (batch, feature) has other_dims elements
    offset = (batch_idx * num_features + feature_idx) * other_dims
    
    # Compute sum of squares
    sum_sq = 0.0
    for i in range(0, other_dims, BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < offset + other_dims
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    
    # Compute mean of squares
    mean_sq = sum_sq / other_dims
    
    # Compute RMS
    rms = tl.sqrt(mean_sq + eps)
    
    # Normalize and store
    for i in range(0, other_dims, BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < offset + other_dims
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = x / rms
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float):
    """
    Applies RMS Normalization using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, *)
        eps: Small value for numerical stability
    
    Returns:
        Output tensor with RMS Normalization applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor dimensions
    batch_size = x.shape[0]
    num_features = x.shape[1]
    
    # Calculate product of remaining dimensions
    other_dims = 1
    for dim in x.shape[2:]:
        other_dims *= dim
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Grid dimensions: [batch_size, num_features]
    grid = (batch_size, num_features)
    
    # Block size - tune for your GPU
    BLOCK_SIZE = 256
    
    # Launch the Triton kernel
    rms_norm_kernel_v2[grid](
        x, out, batch_size, num_features, other_dims, eps,
        BLOCK_SIZE=BLOCK_SIZE
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