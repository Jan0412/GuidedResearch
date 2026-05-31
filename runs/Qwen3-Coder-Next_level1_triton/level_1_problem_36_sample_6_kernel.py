import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    weight_ptr,  # Pointer to scaling parameter (gamma)
    batch_size,  # Batch size
    num_features,  # Number of features
    dim1,  # First spatial dimension
    dim2,  # Second spatial dimension
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Compute total number of feature vectors per sample (dim1 * dim2)
    spatial_size = dim1 * dim2
    
    # Each program handles one batch and one feature index
    batch_id = tl.program_id(0)
    feature_id = tl.program_id(1)
    
    # Calculate base offsets for the current (batch, feature) pair
    base_offset = (batch_id * num_features + feature_id) * spatial_size
    
    # Accumulate sum of squares
    sum_sq = 0.0
    # Loop over spatial dimensions in blocks
    for start in range(0, spatial_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Load values
        x_val = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x_val * x_val, axis=0)
    
    # Compute RMS
    rms = tl.sqrt(sum_sq / spatial_size + eps)
    
    # Normalize and scale
    for start in range(0, spatial_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        x_val = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
        normalized = x_val / rms
        
        # Apply weight (gamma) - since RMSNorm typically doesn't have bias, we just scale
        weight_val = tl.load(weight_ptr + feature_id)
        out_val = normalized * weight_val
        
        tl.store(out_ptr + base_offset + offsets, out_val, mask=mask)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of RMS Normalization.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, dim1, dim2)
        weight: Scaling parameter (gamma) of shape (num_features,)
        eps: Small value to avoid division by zero
    
    Returns:
        Normalized tensor of same shape as x
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, num_features, dim1, dim2 = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Configure grid: (batch_size, num_features)
    grid = (batch_size, num_features)
    
    # Block size for spatial dimension
    BLOCK_SIZE = 256
    
    # Launch the Triton kernel
    rms_norm_kernel[grid](
        x, out, weight,
        batch_size, num_features, dim1, dim2, eps,
        BLOCK_SIZE=BLOCK_SIZE,
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
        # Initialize learnable scaling parameter (gamma)
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, dim1, dim2).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Expand weight to match the expected shape (1, num_features, 1, 1) for broadcasting
        # But our Triton kernel expects weight as (num_features,), so we use it directly
        return triton_rms_norm(x, self.weight, self.eps)