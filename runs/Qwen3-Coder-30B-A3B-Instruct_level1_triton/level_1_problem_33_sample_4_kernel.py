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
    out_ptr,
    N,  # total elements
    C,  # channels
    HxW,  # height * width
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one channel
    channel_id = tl.program_id(0)
    
    if channel_id >= C:
        return
    
    # Calculate offsets for this channel
    channel_offset = channel_id * HxW
    
    # Shared memory for reduction
    mean_shared = tl.shared_ptr(tl.zeros((BLOCK_SIZE,), dtype=tl.float32), shape=(BLOCK_SIZE,))
    var_shared = tl.shared_ptr(tl.zeros((BLOCK_SIZE,), dtype=tl.float32), shape=(BLOCK_SIZE,))
    
    # Load weight and bias for this channel
    weight = tl.load(weight_ptr + channel_id)
    bias = tl.load(bias_ptr + channel_id)
    
    # Load mean and variance for this channel
    mean_val = tl.load(mean_ptr + channel_id)
    var_val = tl.load(var_ptr + channel_id)
    
    # Normalize std dev
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    
    # Process elements in chunks
    for i in range(0, HxW, BLOCK_SIZE):
        # Calculate global offset
        global_offset = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = global_offset < N
        
        # Load input values
        x_vals = tl.load(x_ptr + global_offset, mask=mask, other=0.0)
        
        # Normalize and apply affine transformation
        normalized = (x_vals - mean_val) * inv_std
        out_val = normalized * weight + bias
        
        # Store output
        tl.store(out_ptr + global_offset, out_val, mask=mask)

@triton.jit
def batch_norm_backward_kernel(
    x_ptr,
    grad_out_ptr,
    weight_ptr,
    mean_ptr,
    var_ptr,
    grad_x_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    N,
    C,
    HxW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one channel
    channel_id = tl.program_id(0)
    
    if channel_id >= C:
        return
    
    # Calculate offsets for this channel
    channel_offset = channel_id * HxW
    
    # Load parameters for this channel
    weight = tl.load(weight_ptr + channel_id)
    mean_val = tl.load(mean_ptr + channel_id)
    var_val = tl.load(var_ptr + channel_id)
    
    # Compute inverse standard deviation
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    
    # Initialize accumulators
    sum_grad_weight = 0.0
    sum_grad_bias = 0.0
    
    # Process elements in chunks
    for i in range(0, HxW, BLOCK_SIZE):
        global_offset = channel_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = global_offset < N
        
        # Load values
        x_vals = tl.load(x_ptr + global_offset, mask=mask, other=0.0)
        grad_out_vals = tl.load(grad_out_ptr + global_offset, mask=mask, other=0.0)
        
        # Compute intermediate values
        x_centered = x_vals - mean_val
        normalized = x_centered * inv_std
        
        # Compute gradients
        grad_weight = tl.sum(grad_out_vals * normalized, axis=0)
        grad_bias = tl.sum(grad_out_vals, axis=0)
        
        # Accumulate
        sum_grad_weight += grad_weight
        sum_grad_bias += grad_bias
        
        # Compute gradient w.r.t. input
        grad_x = grad_out_vals * weight * inv_std
        grad_x -= grad_out_vals * tl.sum(grad_out_vals, axis=0) / HxW
        grad_x -= x_centered * tl.sum(grad_out_vals * x_centered, axis=0) / (HxW * var_val)
        grad_x *= inv_std
        
        # Store gradient w.r.t. input
        tl.store(grad_x_ptr + global_offset, grad_x, mask=mask)
    
    # Store accumulated gradients
    if tl.thread_id() == 0:
        tl.atomic_add(grad_weight_ptr + channel_id, sum_grad_weight)
        tl.atomic_add(grad_bias_ptr + channel_id, sum_grad_bias)

def triton_batch_norm_forward(x, weight, bias, mean, var, eps=1e-5):
    """Forward pass for batch normalization using Triton"""
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda, "Weight tensor must be on CUDA"
    assert bias.is_cuda, "Bias tensor must be on CUDA"
    assert mean.is_cuda, "Mean tensor must be on CUDA"
    assert var.is_cuda, "Variance tensor must be on CUDA"
    
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    
    # Get dimensions
    N = x.numel()
    C = weight.shape[0]  # channels
    HxW = N // C  # height * width
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = lambda meta: (C,)
    
    # Launch kernel
    batch_norm_forward_kernel[grid](
        x_ptr=x.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr(),
        mean_ptr=mean.data_ptr(),
        var_ptr=var.data_ptr(),
        out_ptr=out.data_ptr(),
        N=N,
        C=C,
        HxW=HxW,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model with custom Triton kernels for Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        # For inference mode, use running statistics
        if not self.training:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var
        else:
            # Compute batch statistics
            batch_mean = x.mean(dim=[0, 2, 3])  # Mean over batch, height, width
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)  # Variance over batch, height, width
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            mean = batch_mean
            var = batch_var
            
        # Apply batch norm using Triton kernel
        return triton_batch_norm_forward(x, self.weight, self.bias, mean, var, self.eps)