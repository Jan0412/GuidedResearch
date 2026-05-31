import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batch_norm_kernel(
    x_ptr, y_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N, C, H, W, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid dimensions: (N, C)
    n_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    
    # Calculate base offset for the current batch and channel
    # Tensor shape is (N, C, H, W), contiguous memory layout
    # Offset = n * (C * H * W) + c * (H * W)
    base_offset = n_idx * C * H * W + c_idx * H * W
    
    # Load channel-wise statistics and parameters
    # These are constant for the entire spatial block of this channel
    mean = tl.load(mean_ptr + c_idx)
    var = tl.load(var_ptr + c_idx)
    weight = tl.load(weight_ptr + c_idx)
    bias = tl.load(bias_ptr + c_idx)
    
    # Precompute normalization terms to minimize per-element operations
    # y = (x - mean) / sqrt(var + eps) * weight + bias
    # y = x * (weight / sqrt(var + eps)) + (bias - mean * weight / sqrt(var + eps))
    inv_std = tl.rsqrt(var + eps)
    scale = weight * inv_std
    shift = bias - mean * scale
    
    # Generate offsets for the spatial dimensions
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (H * W)
    
    # Load input tensor block
    x = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
    
    # Apply normalization, scaling, and shifting
    y = x * scale + shift
    
    # Store result
    tl.store(y_ptr + base_offset + offsets, y, mask=mask)


def triton_batch_norm(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, 
                      weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Wrapper function to launch the custom Triton Batch Normalization kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    
    # The block size covers the entire spatial dimension for each channel
    # This allows loading statistics once per block and vectorized access to spatial data
    BLOCK_SIZE = H * W
    
    # Grid configuration: one block per (batch, channel) pair
    grid = (N, C)
    
    batch_norm_kernel[grid](
        x, y, mean, var, weight, bias,
        N, C, H, W, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for Batch Normalization.
    """
    def __init__(self, num_features: int):
        super().__init__()
        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Initialize running statistics (buffers)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use custom Triton kernel for inference-style batch normalization
        # This replaces the nn.BatchNorm2d operator with a fused, optimized kernel
        return triton_batch_norm(x, self.running_mean, self.running_var, 
                                 self.weight, self.bias, self.eps)