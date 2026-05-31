import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    X,  # Input tensor pointer
    Y,  # Output tensor pointer
    W,  # Weight (scale) parameter pointer
    B,  # Bias (shift) parameter pointer
    N,  # Total number of elements in the normalized dimensions
    C,  # Number of normalized dimensions (product of normalized_shape)
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one sample in the batch
    batch_idx = tl.program_id(0)
    
    # Calculate offset to the start of this sample's normalized region
    sample_offset = batch_idx * C
    
    # Compute mean
    mean_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start_n in range(0, C, BLOCK_SIZE):
        offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = offsets < C
        x = tl.load(X + sample_offset + offsets, mask=mask, other=0.0)
        mean_sum += x.to(tl.float32)
    
    mean = tl.sum(mean_sum) / C
    
    # Compute variance
    var_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start_n in range(0, C, BLOCK_SIZE):
        offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = offsets < C
        x = tl.load(X + sample_offset + offsets, mask=mask, other=0.0)
        x_shifted = x - mean
        var_sum += x_shifted * x_shifted
    
    var = tl.sum(var_sum) / C
    std = tl.sqrt(var + eps)
    
    # Normalize and apply scale/bias
    for start_n in range(0, C, BLOCK_SIZE):
        offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = offsets < C
        x = tl.load(X + sample_offset + offsets, mask=mask, other=0.0)
        w = tl.load(W + offsets, mask=mask, other=0.0)
        b = tl.load(B + offsets, mask=mask, other=0.0)
        
        # Normalize
        x_norm = (x - mean) / std
        
        # Apply scale and bias
        out = x_norm * w + b
        
        tl.store(Y + sample_offset + offsets, out, mask=mask)


def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton-based LayerNorm implementation.
    
    Args:
        x: Input tensor of shape (batch_size, *normalized_shape)
        weight: Scale parameter of shape normalized_shape
        bias: Bias parameter of shape normalized_shape
        eps: Epsilon for numerical stability
    
    Returns:
        Output tensor of same shape as x
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get shapes
    batch_size = x.shape[0]
    normalized_shape = weight.shape
    
    # Calculate total normalized dimension size
    C = 1
    for s in normalized_shape:
        C *= s
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Grid: one block per batch element
    grid = lambda meta: (batch_size,)
    
    # Launch kernel
    layernorm_kernel[grid](
        x, out, weight, bias,
        batch_size * C, C, eps,
        BLOCK_SIZE=256
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using Triton kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with Triton optimization.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        # Create parameters for LayerNorm
        self.normalized_shape = normalized_shape
        C = 1
        for s in normalized_shape:
            C *= s
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization using Triton kernel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return triton_layernorm(x, self.weight, self.bias, self.eps)