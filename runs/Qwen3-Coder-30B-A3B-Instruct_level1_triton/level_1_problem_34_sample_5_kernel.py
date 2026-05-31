import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def instance_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    batch_size,
    channels,
    height,
    width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Shared memory for reduction
    shared_mean = tl.shared_ptr(mean_ptr, BLOCK_SIZE)
    shared_var = tl.shared_ptr(var_ptr, BLOCK_SIZE)
    
    # Calculate offsets
    batch_offset = batch_idx * channels * height * width
    channel_offset = channel_idx * height * width
    
    # Load weights and biases
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Compute mean and variance
    mean = 0.0
    var = 0.0
    
    # First pass: compute mean
    for i in range(0, height * width, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < height * width
        
        if mask.any():
            x_vals = tl.load(x_ptr + batch_offset + channel_offset + idx, mask=mask, other=0.0)
            mean += tl.sum(x_vals)
    
    mean = mean / (height * width)
    
    # Second pass: compute variance
    for i in range(0, height * width, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < height * width
        
        if mask.any():
            x_vals = tl.load(x_ptr + batch_offset + channel_offset + idx, mask=mask, other=0.0)
            diff = x_vals - mean
            var += tl.sum(diff * diff)
    
    var = var / (height * width)
    
    # Normalize and apply affine transformation
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    for i in range(0, height * width, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < height * width
        
        if mask.any():
            x_vals = tl.load(x_ptr + batch_offset + channel_offset + idx, mask=mask, other=0.0)
            normalized = (x_vals - mean) * inv_std
            output = normalized * weight + bias
            tl.store(out_ptr + batch_offset + channel_offset + idx, output, mask=mask)

def triton_instance_norm(x, weight, bias, eps=1e-5):
    """
    Triton implementation of Instance Normalization
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda and bias.is_cuda, "Weight and bias must be on CUDA"
    
    batch_size, channels, height, width = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Prepare mean and var tensors (for intermediate computation)
    mean = torch.empty(batch_size, channels, device=x.device, dtype=torch.float32)
    var = torch.empty(batch_size, channels, device=x.device, dtype=torch.float32)
    
    # Grid configuration
    grid = (batch_size, channels)
    BLOCK_SIZE = 1024
    
    # Launch kernel
    instance_norm_kernel[grid](
        x, weight, bias, out, mean, var,
        batch_size, channels, height, width,
        eps, BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer with custom Triton kernel.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        # Convert to float32 for computation
        if x.dtype != torch.float32:
            original_dtype = x.dtype
            x = x.float()
            weight = self.weight.float()
            bias = self.bias.float()
        else:
            original_dtype = None
            weight = self.weight
            bias = self.bias
            
        # Apply custom Triton kernel
        out = triton_instance_norm(x, weight, bias, self.eps)
        
        # Convert back to original dtype if needed
        if original_dtype is not None:
            out = out.to(original_dtype)
            
        return out