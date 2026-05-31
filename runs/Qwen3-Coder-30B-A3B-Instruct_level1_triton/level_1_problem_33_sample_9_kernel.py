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
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Each program processes one channel
    if row_idx >= C:
        return
    
    # Calculate the starting position for this channel
    channel_offset = row_idx * H * W
    
    # Shared memory for reduction operations
    mean_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    var_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load data for this channel
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * H * W
        
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute mean
        mean_val += x_vals
        
    # Reduce to get mean for this channel
    mean_sum = tl.sum(mean_val, axis=0)
    mean = mean_sum / (H * W)
    
    # Compute variance
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * H * W
        
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        diff = x_vals - mean
        var_val += diff * diff
    
    # Reduce to get variance for this channel
    var_sum = tl.sum(var_val, axis=0)
    var = var_sum / (H * W)
    
    # Store mean and variance for this channel
    tl.store(mean_ptr + row_idx, mean)
    tl.store(var_ptr + row_idx, var)
    
    # Normalize and apply affine transformation
    weight = tl.load(weight_ptr + row_idx)
    bias = tl.load(bias_ptr + row_idx)
    
    inv_std = tl.math.rsqrt(var + eps)
    
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * H * W
        
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        normalized = (x_vals - mean) * inv_std
        output = normalized * weight + bias
        
        tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def batch_norm_backward_kernel(
    x_ptr,
    grad_output_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    grad_x_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Each program processes one channel
    if row_idx >= C:
        return
    
    # Calculate the starting position for this channel
    channel_offset = row_idx * H * W
    
    # Load statistics for this channel
    mean = tl.load(mean_ptr + row_idx)
    var = tl.load(var_ptr + row_idx)
    weight = tl.load(weight_ptr + row_idx)
    
    # Compute inverse standard deviation
    inv_std = tl.math.rsqrt(var + eps)
    
    # Compute gradients for weight and bias
    weight_grad_sum = tl.zeros((1,), dtype=tl.float32)
    bias_grad_sum = tl.zeros((1,), dtype=tl.float32)
    
    # First pass: compute sum of gradients for weight and bias
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * H * W
        
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        grad_output_vals = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize input
        normalized = (x_vals - mean) * inv_std
        
        # Accumulate gradients
        weight_grad_sum += tl.sum(grad_output_vals * normalized)
        bias_grad_sum += tl.sum(grad_output_vals)
    
    # Store gradients for weight and bias
    tl.store(grad_weight_ptr + row_idx, weight_grad_sum)
    tl.store(grad_bias_ptr + row_idx, bias_grad_sum)
    
    # Second pass: compute gradient w.r.t. input
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * H * W
        
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        grad_output_vals = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
        
        # Compute intermediate values
        normalized = (x_vals - mean) * inv_std
        dgamma = tl.sum(grad_output_vals * normalized)
        dbeta = tl.sum(grad_output_vals)
        
        # Gradient w.r.t. input
        dx = (grad_output_vals - dbeta / (H * W) - normalized * dgamma / (H * W)) * weight * inv_std
        
        tl.store(grad_x_ptr + offsets, dx, mask=mask)

class TritonBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(TritonBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # x shape: (N, C, H, W)
        N, C, H, W = x.shape
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        if self.training:
            # Use Triton kernel for training
            output = torch.empty_like(x)
            
            # Allocate buffers for mean and var
            mean = torch.empty(C, dtype=torch.float32, device=x.device)
            var = torch.empty(C, dtype=torch.float32, device=x.device)
            
            # Grid configuration
            BLOCK_SIZE = 1024
            
            # Launch kernel for computing mean and variance
            grid_mean = lambda meta: (C,)
            batch_norm_forward_kernel[grid_mean](
                x, 
                self.weight.data, 
                self.bias.data, 
                mean, 
                var, 
                output, 
                N, C, H, W, 
                self.eps, 
                BLOCK_SIZE
            )
            
            # Update running stats
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                    
            return output
        else:
            # Use PyTorch's native implementation for inference
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias, 
                self.training, self.momentum, self.eps
            )

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer with Triton optimizations.

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