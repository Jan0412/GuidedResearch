import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    X,  # Pointer to input tensor
    Y,  # Pointer to output tensor
    M,  # Number of features (normalized dimension)
    N,  # Total number of elements in non-feature dimensions
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
    FEATURES_BLOCK: tl.constexpr
):
    # Each program handles a subset of the non-feature dimensions
    batch_idx = tl.program_id(0)
    
    # Compute the mean of squares for this sample
    sum_sq = 0.0
    for i in range(0, M, FEATURES_BLOCK):
        # Load features for this block
        offsets = batch_idx * M + i + tl.arange(0, FEATURES_BLOCK)
        mask = (i + tl.arange(0, FEATURES_BLOCK)) < M
        x = tl.load(X + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    
    # Compute RMS: sqrt(mean + eps)
    mean_sq = sum_sq / M
    rms = tl.sqrt(mean_sq + eps)
    
    # Normalize and store
    for i in range(0, M, FEATURES_BLOCK):
        offsets = batch_idx * M + i + tl.arange(0, FEATURES_BLOCK)
        mask = (i + tl.arange(0, FEATURES_BLOCK)) < M
        x = tl.load(X + offsets, mask=mask, other=0.0)
        y = x / rms
        tl.store(Y + offsets, y, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float = 1e-5):
    """
    Applies RMS Normalization using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, *)
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized tensor with same shape as input
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    batch_size = x.shape[0]
    num_features = x.shape[1]
    other_dims = x.shape[2:]
    
    # Calculate total non-feature elements
    N = 1
    for d in other_dims:
        N *= d
    
    total_elements = batch_size * num_features * N
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Reshape to 2D for easier processing: (batch_size, num_features * other_dims)
    x_2d = x.view(batch_size, -1)
    out_2d = out.view(batch_size, -1)
    
    # Configure kernel launch parameters
    # Each program handles one batch (one set of features across all other dimensions)
    grid = (batch_size,)
    
    # Use a reasonable block size for feature dimension
    FEATURES_BLOCK = 32
    
    # Launch kernel
    rms_norm_kernel[grid](
        x_2d, out_2d,
        num_features, N,
        eps,
        BLOCK_SIZE=128,
        FEATURES_BLOCK=FEATURES_BLOCK
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_rms_norm(x, self.eps)