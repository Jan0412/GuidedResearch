import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def instance_norm_kernel(
    X_ptr,  # Input tensor pointer
    Y_ptr,  # Output tensor pointer
    Gamma_ptr,  # Scale parameter pointer
    Beta_ptr,  # Shift parameter pointer
    N,  # Total number of elements (batch * channels * height * width)
    C,  # Number of channels
    spatial_size,  # height * width for each channel
    eps,  # Small value for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample-channel pair
    sample_idx = tl.program_id(0) // C
    channel_idx = tl.program_id(0) % C
    
    # Calculate starting offset for this sample-channel
    start_offset = (sample_idx * C + channel_idx) * spatial_size
    
    # Compute mean
    mean = 0.0
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = start_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (start_offset + spatial_size)
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
        mean += tl.sum(x * mask, axis=0)
    
    # Normalize mean calculation
    mean = mean / spatial_size
    
    # Compute variance
    var = 0.0
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = start_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (start_offset + spatial_size)
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
        var += tl.sum((x - mean) * (x - mean) * mask, axis=0)
    
    var = var / spatial_size
    
    # Compute standard deviation with epsilon for stability
    std = tl.sqrt(var + eps)
    
    # Load gamma and beta (scale and shift parameters)
    gamma = tl.load(Gamma_ptr + channel_idx) if Gamma_ptr is not None else 1.0
    beta = tl.load(Beta_ptr + channel_idx) if Beta_ptr is not None else 0.0
    
    # Normalize and apply affine transformation
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = start_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (start_offset + spatial_size)
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
        normalized = (x - mean) / std
        out = normalized * gamma + beta
        tl.store(Y_ptr + offsets, out, mask=mask)

def triton_instance_norm(x, weight, bias, eps=1e-5):
    """
    Triton implementation of InstanceNorm2d.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, height, width)
        weight: Scale parameter (gamma)
        bias: Shift parameter (beta)
        eps: Small value for numerical stability
    
    Returns:
        Normalized output tensor
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, num_features, height, width = x.shape
    spatial_size = height * width
    N = batch_size * num_features
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Create grid: one program per (sample, channel) pair
    grid = (N,)
    
    # Launch kernel
    instance_norm_kernel[grid](
        x, out, weight, bias, 
        x.numel(), num_features, spatial_size, eps,
        BLOCK_SIZE=128
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using custom Triton kernel.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer with Triton kernel implementation.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        # Register buffers for weight and bias (gamma and beta)
        # These are equivalent to the learnable parameters in nn.InstanceNorm2d
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5  # Default epsilon value from PyTorch
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization using Triton kernel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x, self.weight, self.bias, self.eps)