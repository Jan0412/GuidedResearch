import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    weight_ptr,  # Pointer to weight tensor (gamma)
    n_elements,  # Total number of elements in input/output
    num_features,  # Number of features (C)
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
    FEATURES_BLOCK: tl.constexpr,
):
    # Each program handles one batch sample at a time
    batch_id = tl.program_id(0)
    
    # Calculate base pointer offset for this batch
    batch_offset = batch_id * num_features * n_elements // batch_id if batch_id == 0 else 0
    # Recalculate properly
    # Actually n_elements is total, so per-batch size is n_elements // batch_size
    # Let's recalculate properly
    
    # Get the number of elements per batch
    batch_size = tl.load(tl.make_tensor_ptr([batch_size], dtype=tl.int32))  # We'll pass this as a separate tensor
    
    # Since Triton doesn't support dynamic batch_size from tensor in simple way, 
    # let's restructure: we'll launch grid over (batch_size, *spatial_dims), but that's complex.
    # Instead, let's do batched processing in a different way.
    
    # For simplicity and performance, we'll launch grid over batch and then handle features in blocks.
    # But the kernel above is too complex. Let's simplify by launching grid over (batch * spatial_elements)
    # and handle feature dimension within each program.
    
    # Let's use a different approach: launch grid over (batch_size * prod(spatial_dims))
    # and process the feature dimension in each program
    
    # Get current spatial index
    spatial_idx = tl.program_id(0)
    
    # Calculate batch index and spatial position within batch
    # spatial_size = n_elements // (batch_size * num_features)
    # batch_idx = spatial_idx // spatial_size
    # spatial_pos = spatial_idx % spatial_size
    
    # But we don't have spatial_size directly. Let's compute it.
    # Actually, let's redesign the grid to be (batch_size * prod(spatial_dims))
    # and handle feature dimension in the kernel
    
    # Since we know the shape in the wrapper, we can pass spatial_size
    # Let's add spatial_size as a parameter
    
    # Get batch index and feature index
    batch_idx = spatial_idx // (n_elements // batch_size)
    spatial_pos = spatial_idx % (n_elements // batch_size)
    
    # But this is getting messy. Let's use a better approach.
    # Launch grid over (batch_size, spatial_size), and handle features in a loop
    # Actually, for simplicity, we'll use the standard approach: process one batch at a time
    
    # Let's launch grid over (batch_size * prod(spatial_dims))
    # and process features in each program
    
    # Get the number of elements per feature per batch
    # elements_per_feature = n_elements // (batch_size * num_features)
    
    # Since we're processing one (batch, spatial) position at a time,
    # we need to collect all features for that position, compute RMS, then normalize
    
    # But this is inefficient for large feature dimensions. Let's do better.
    
    # Better approach: launch grid over (batch_size * prod(spatial_dims))
    # and process features in blocks with a loop
    
    # Since the feature dimension is small (64 in the example), we can process all features in one go
    
    # Get the base pointer for this batch and spatial position
    # elements_per_feature = n_elements // (batch_size * num_features)
    # offset = batch_idx * (num_features * elements_per_feature) + spatial_pos
    
    # Actually, let's use a cleaner approach: we'll pass the strides
    
    # For simplicity, let's assume contiguous memory and calculate strides
    
    # Calculate strides for (batch, feature, spatial)
    # stride_batch = num_features * elements_per_feature
    # stride_feature = elements_per_feature
    
    # But we don't have elements_per_feature directly. Let's compute it from n_elements
    
    # Let's simplify: we'll assume the input is (batch_size, num_features, *)
    # and calculate the spatial size from n_elements and batch_size, num_features
    
    # n_elements = batch_size * num_features * spatial_size
    # spatial_size = n_elements // (batch_size * num_features)
    
    # But batch_size is a compile-time constant? No, it's runtime.
    
    # Let's redesign: we'll launch grid over (batch_size * prod(spatial_dims))
    # and process features in a loop
    
    # Get the spatial index within the batch
    # batch_elements = n_elements // batch_size
    # spatial_idx_in_batch = spatial_idx % batch_elements
    
    # Get the feature index: feature_idx = spatial_idx_in_batch // spatial_size
    # spatial_pos_in_feature = spatial_idx_in_batch % spatial_size
    
    # But again, we need spatial_size. Let's compute it.
    # spatial_size = n_elements // (batch_size * num_features)
    
    # Actually, let's pass spatial_size as a parameter
    
    # Let's rewrite the kernel with better structure
    pass  # Placeholder for now - we'll rewrite the whole kernel below


@triton.jit
def rms_norm_kernel_v2(
    x_ptr,  # Pointer to input tensor [batch_size, num_features, *]
    out_ptr,  # Pointer to output tensor
    weight_ptr,  # Pointer to weight tensor [num_features]
    batch_size,  # Batch size
    num_features,  # Number of features
    spatial_size,  # Total spatial size (prod of all spatial dims)
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Launch grid over (batch_size * spatial_size)
    # Each program handles one (batch, spatial) position and processes all features
    
    # Get the program index
    pid = tl.program_id(0)
    
    # Calculate batch and spatial indices
    batch_idx = pid // spatial_size
    spatial_idx = pid % spatial_size
    
    # Calculate the base offset for this (batch, spatial) position
    # Assuming input is [batch_size, num_features, spatial_size_per_batch]
    # where spatial_size_per_batch = spatial_size
    # So total elements per batch = num_features * spatial_size
    base_offset = batch_idx * num_features * spatial_size + spatial_idx
    
    # We'll collect all features for this position and compute RMS
    # Since num_features is typically small, we can process all features in one go
    
    # Compute sum of squares
    sum_sq = 0.0
    for f in range(num_features):
        offset = base_offset + f * spatial_size
        x_val = tl.load(x_ptr + offset)
        sum_sq += x_val * x_val
    
    # Compute RMS
    rms = tl.sqrt(sum_sq / num_features + eps)
    
    # Normalize and store
    for f in range(num_features):
        offset = base_offset + f * spatial_size
        x_val = tl.load(x_ptr + offset)
        normalized = x_val / rms
        # Apply weight (gamma)
        weight = tl.load(weight_ptr + f)
        out_val = normalized * weight
        tl.store(out_ptr + offset, out_val)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """
    Applies RMS Normalization using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, *).
        weight: Weight tensor of shape (num_features,).
        eps: Epsilon for numerical stability.
        
    Returns:
        Output tensor with RMS Normalization applied.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert weight.is_cuda, "Weight tensor must be on CUDA."
    
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get shape information
    batch_size = x.shape[0]
    num_features = x.shape[1]
    spatial_size = x.numel() // (batch_size * num_features)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Grid: batch_size * spatial_size
    grid = (batch_size * spatial_size,)
    
    # Launch the Triton kernel
    BLOCK_SIZE = 128  # Not really used in this kernel structure, but kept for consistency
    
    rms_norm_kernel_v2[grid](
        x, out, weight,
        batch_size, num_features, spatial_size, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using Triton kernel.
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
        # Initialize weight (gamma) parameter
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_rms_norm(x, self.weight, self.eps)