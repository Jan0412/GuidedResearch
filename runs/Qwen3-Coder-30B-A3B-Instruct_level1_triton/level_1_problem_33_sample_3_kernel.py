import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def batch_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    output_ptr,
    N,
    C,
    H,
    W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(0)
    
    # Each program processes one channel
    if pid >= C:
        return
    
    # Calculate offset for this channel
    channel_offset = pid * H * W
    
    # Shared memory for reduction operations
    mean_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    var_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute mean for this channel
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H * W
        
        x_vals = tl.load(x_ptr + channel_offset + offsets, mask=mask, other=0.0)
        
        # Accumulate sum for mean calculation
        mean_val += x_vals
    
    # Reduce to get mean
    mean_sum = tl.sum(mean_val, axis=0)
    mean = mean_sum / (H * W)
    
    # Compute variance
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H * W
        
        x_vals = tl.load(x_ptr + channel_offset + offsets, mask=mask, other=0.0)
        
        # Compute squared differences from mean
        diff = x_vals - mean
        var_val += diff * diff
    
    # Reduce to get variance
    var_sum = tl.sum(var_val, axis=0)
    var = var_sum / (H * W)
    
    # Normalize and apply scale and shift
    inv_std = tl.math.rsqrt(var + eps)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + pid, mask=True, other=1.0)
    bias = tl.load(bias_ptr + pid, mask=True, other=0.0)
    
    # Apply normalization and affine transformation
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H * W
        
        x_vals = tl.load(x_ptr + channel_offset + offsets, mask=mask, other=0.0)
        
        # Normalize
        normalized = (x_vals - mean) * inv_std
        
        # Scale and shift
        output_val = normalized * weight + bias
        
        # Store result
        tl.store(output_ptr + channel_offset + offsets, output_val, mask=mask)

@triton.jit
def batch_norm_backward_kernel(
    x_ptr,
    grad_output_ptr,
    weight_ptr,
    mean_ptr,
    var_ptr,
    grad_x_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    N,
    C,
    H,
    W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= C:
        return
    
    channel_offset = pid * H * W
    
    # Load parameters
    weight = tl.load(weight_ptr + pid, mask=True, other=1.0)
    mean = tl.load(mean_ptr + pid, mask=True, other=0.0)
    var = tl.load(var_ptr + pid, mask=True, other=1.0)
    
    inv_std = tl.math.rsqrt(var + eps)
    
    # Initialize accumulators
    sum_grad_output = 0.0
    sum_grad_output_x_centered = 0.0
    
    # Compute sums for gradients
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H * W
        
        x_vals = tl.load(x_ptr + channel_offset + offsets, mask=mask, other=0.0)
        grad_output_vals = tl.load(grad_output_ptr + channel_offset + offsets, mask=mask, other=0.0)
        
        sum_grad_output += tl.sum(grad_output_vals)
        sum_grad_output_x_centered += tl.sum(grad_output_vals * (x_vals - mean))
    
    # Compute gradients
    grad_bias = sum_grad_output
    grad_weight = sum_grad_output_x_centered * inv_std
    
    # Store gradients
    tl.store(grad_bias_ptr + pid, grad_bias)
    tl.store(grad_weight_ptr + pid, grad_weight)
    
    # Compute gradient w.r.t input
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H * W
        
        x_vals = tl.load(x_ptr + channel_offset + offsets, mask=mask, other=0.0)
        grad_output_vals = tl.load(grad_output_ptr + channel_offset + offsets, mask=mask, other=0.0)
        
        # Compute gradient w.r.t input
        grad_x = (grad_output_vals * weight * inv_std) - \
                 (sum_grad_output / (H * W)) * (weight * inv_std) - \
                 (sum_grad_output_x_centered * inv_std * (x_vals - mean) / (H * W)) * (weight * inv_std)
        
        tl.store(grad_x_ptr + channel_offset + offsets, grad_x, mask=mask)

class TritonBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(TritonBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if not x.is_cuda:
            return F.batch_norm(x, self.running_mean, self.running_var, 
                              self.weight, self.bias, self.training, self.momentum, self.eps)
        
        N, C, H, W = x.shape
        
        # Ensure proper alignment
        x = x.contiguous()
        
        # Allocate output
        output = torch.empty_like(x)
        
        # Configure kernel launch parameters
        BLOCK_SIZE = 1024
        grid_size = (C + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        batch_norm_forward_kernel[grid_size](
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            output,
            N,
            C,
            H,
            W,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer with Triton implementation.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.bn = TritonBatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(x)