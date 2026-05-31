import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    X_ptr,           # Input tensor pointer: [B, C, H, W]
    Y_ptr,           # Output tensor pointer: [B, C, H, W]
    mean_ptr,        # Mean tensor pointer: [B, C] (optional, for inference)
    var_ptr,         # Variance tensor pointer: [B, C] (optional, for inference)
    weight_ptr,      # Scale parameter pointer: [C]
    bias_ptr,        # Shift parameter pointer: [C]
    B, C, H, W,      # Dimensions
    eps,             # Epsilon for numerical stability
    BLOCK_SIZE_HW: tl.constexpr,  # Block size for H*W dimension
    BLOCK_SIZE_C: tl.constexpr,   # Block size for C dimension (if needed)
):
    # Each program processes one (batch, channel) pair
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate offsets for this (batch, channel) pair
    # We'll process H*W elements at a time
    hw_total = H * W
    
    # Compute mean and variance for this (batch, channel) pair
    # Load all H*W values for this (batch, channel) pair
    # We'll do this in multiple steps if needed
    sum_val = tl.zeros((1,), dtype=tl.float32)
    sum_sq_val = tl.zeros((1,), dtype=tl.float32)
    
    for i in range(0, hw_total, BLOCK_SIZE_HW):
        offsets = i + tl.arange(0, BLOCK_SIZE_HW)
        mask = offsets < hw_total
        
        # Calculate the actual index in the flattened tensor
        idx = batch_idx * (C * H * W) + channel_idx * (H * W) + offsets
        
        # Load values
        x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq_val += tl.sum(x * x, axis=0)
    
    # Compute mean and variance
    hw_float = hw_total.astype(tl.float32)
    mean = sum_val / hw_float
    var = sum_sq_val / hw_float - mean * mean
    
    # Store mean and variance if needed (for inference)
    if mean_ptr is not None:
        tl.store(mean_ptr + batch_idx * C + channel_idx, mean)
    if var_ptr is not None:
        tl.store(var_ptr + batch_idx * C + channel_idx, var)
    
    # Compute normalized output
    std = tl.sqrt(var + eps)
    
    # Load weight and bias for this channel
    w = tl.load(weight_ptr + channel_idx).to(tl.float32) if weight_ptr is not None else 1.0
    b = tl.load(bias_ptr + channel_idx).to(tl.float32) if bias_ptr is not None else 0.0
    
    # Normalize and scale/shift
    for i in range(0, hw_total, BLOCK_SIZE_HW):
        offsets = i + tl.arange(0, BLOCK_SIZE_HW)
        mask = offsets < hw_total
        
        # Calculate the actual index in the flattened tensor
        idx = batch_idx * (C * H * W) + channel_idx * (H * W) + offsets
        
        # Load values
        x = tl.load(X_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize: (x - mean) / std * weight + bias
        x_norm = (x - mean) / std * w + b
        
        # Store result
        tl.store(Y_ptr + idx, x_norm.to(X_ptr.dtype.element_ty), mask=mask)


def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None, eps: float = 1e-5):
    """
    Triton implementation of Instance Normalization for FP32 tensors.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Scale parameter of shape (C,)
        bias: Shift parameter of shape (C,)
        eps: Small value for numerical stability
    
    Returns:
        Normalized tensor of same shape as input
    """
    assert x.is_cuda and x.dtype == torch.float32, "Tensors must be on CUDA and FP32."
    x = x.contiguous()
    
    B, C, H, W = x.shape
    
    # Prepare output tensor
    y = torch.empty_like(x)
    
    # Prepare weight and bias if not provided (use ones and zeros)
    if weight is None:
        weight = torch.ones(C, device=x.device, dtype=torch.float32)
    else:
        weight = weight.to(torch.float32).contiguous()
    
    if bias is None:
        bias = torch.zeros(C, device=x.device, dtype=torch.float32)
    else:
        bias = bias.to(torch.float32).contiguous()
    
    # Determine grid dimensions: [B, C] for batch and channel processing
    grid = (B, C)
    
    # Set block size for H*W dimension
    BLOCK_SIZE_HW = 256
    
    # Launch the kernel
    instance_norm_kernel[grid](
        x, y, None, None,  # mean_ptr, var_ptr (None for training)
        weight, bias,
        B, C, H, W,
        eps,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        BLOCK_SIZE_C=1,  # Not used but kept for interface completeness
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer with optimized Triton implementation.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        # Register buffers for weight and bias to match nn.InstanceNorm2d interface
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x, self.weight, self.bias, self.eps)