import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def instance_norm_forward_kernel(
    X_ptr,  # Input tensor: (B, C, H, W)
    Y_ptr,  # Output tensor: (B, C, H, W)
    mean_ptr,  # Mean per channel per sample: (B, C)
    rstd_ptr,  # Reciprocal std per channel per sample: (B, C)
    weight_ptr,  # Scale parameter: (C,)
    bias_ptr,  # Shift parameter: (C,)
    B, C, H, W,
    eps,
    stride_b, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one (batch, channel) pair
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)

    # Compute mean and variance for this (batch, channel) pair
    start_offset = batch_idx * stride_b + channel_idx * stride_c
    
    # Accumulate for mean
    sum_val = 0.0
    sum_sq_val = 0.0
    
    # Loop over H*W elements
    for hw_idx in range(H * W):
        offset = start_offset + (hw_idx // W) * stride_h + (hw_idx % W) * stride_w
        x = tl.load(X_ptr + offset)
        sum_val += x
        sum_sq_val += x * x
    
    # Compute mean and variance
    n = H * W
    mean = sum_val / n
    var = sum_sq_val / n - mean * mean
    
    # Compute reciprocal standard deviation (with numerical stability)
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and rstd for backward pass (if needed, but we'll compute on-the-fly for forward)
    tl.store(mean_ptr + batch_idx * C + channel_idx, mean)
    tl.store(rstd_ptr + batch_idx * C + channel_idx, rstd)
    
    # Normalize and apply affine transform
    weight = tl.load(weight_ptr + channel_idx) if weight_ptr is not None else 1.0
    bias = tl.load(bias_ptr + channel_idx) if bias_ptr is not None else 0.0
    
    for hw_idx in range(H * W):
        offset = start_offset + (hw_idx // W) * stride_h + (hw_idx % W) * stride_w
        x = tl.load(X_ptr + offset)
        normalized = (x - mean) * rstd
        out = normalized * weight + bias
        tl.store(Y_ptr + offset, out)


@triton.jit
def instance_norm_backward_kernel(
    dY_ptr,  # Gradient of loss w.r.t. output
    X_ptr,   # Input tensor
    mean_ptr,  # Mean per channel per sample: (B, C)
    rstd_ptr,  # Reciprocal std per channel per sample: (B, C)
    weight_ptr,  # Scale parameter: (C,)
    dB_ptr,  # Gradient w.r.t. bias: (C,)
    dW_ptr,  # Gradient w.r.t. weight: (C,)
    B, C, H, W,
    stride_b, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one (batch, channel) pair
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)

    # Load precomputed stats
    mean = tl.load(mean_ptr + batch_idx * C + channel_idx)
    rstd = tl.load(rstd_ptr + batch_idx * C + channel_idx)
    weight = tl.load(weight_ptr + channel_idx) if weight_ptr is not None else 1.0
    
    # Accumulate gradients
    dnorm_sum = 0.0
    dnorm_x_sum = 0.0
    
    start_offset = batch_idx * stride_b + channel_idx * stride_c
    
    for hw_idx in range(H * W):
        offset = start_offset + (hw_idx // W) * stride_h + (hw_idx % W) * stride_w
        dy = tl.load(dY_ptr + offset)
        x = tl.load(X_ptr + offset)
        
        # Gradient w.r.t. normalized input
        dnorm = dy * weight
        dnorm_sum += dnorm
        dnorm_x_sum += dnorm * (x - mean)
    
    n = H * W
    # Gradient w.r.t. input x
    for hw_idx in range(H * W):
        offset = start_offset + (hw_idx // W) * stride_h + (hw_idx % W) * stride_w
        dy = tl.load(dY_ptr + offset)
        x = tl.load(X_ptr + offset)
        dnorm = dy * weight
        dx = (dnorm - (dnorm_sum + (x - mean) * dnorm_x_sum / n) / n) * rstd
        tl.store(dY_ptr + offset, dx)  # Store gradient in dY_ptr (reusing memory)
    
    # Accumulate weight and bias gradients (done in separate kernel for simplicity)
    if weight_ptr is not None:
        dW = dnorm_x_sum * rstd
        tl.atomic_add(dW_ptr + channel_idx, dW)
    d_b = dnorm_sum * rstd
    tl.atomic_add(dB_ptr + channel_idx, d_b)


class InstanceNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        B, C, H, W = x.shape
        x = x.contiguous()
        
        # Allocate output tensors
        y = torch.empty_like(x)
        mean = torch.empty(B, C, device=x.device, dtype=x.dtype)
        rstd = torch.empty(B, C, device=x.device, dtype=x.dtype)
        
        # Determine strides
        stride_b = x.stride(0)
        stride_c = x.stride(1)
        stride_h = x.stride(2)
        stride_w = x.stride(3)
        
        # Grid: (B, C)
        grid = (B, C)
        
        # Launch kernel
        instance_norm_forward_kernel[grid](
            x, y, mean, rstd,
            weight, bias,
            B, C, H, W,
            eps,
            stride_b, stride_c, stride_h, stride_w,
            BLOCK_SIZE=256,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.eps = eps
        ctx.B, ctx.C, ctx.H, ctx.W = B, C, H, W
        ctx.strides = (stride_b, stride_c, stride_h, stride_w)
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mean, rstd = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        B, C, H, W = ctx.B, ctx.C, ctx.H, ctx.W
        eps = ctx.eps
        stride_b, stride_c, stride_h, stride_w = ctx.strides
        
        # Allocate gradients
        grad_weight = torch.zeros(C, device=x.device, dtype=x.dtype) if weight is not None else None
        grad_bias = torch.zeros(C, device=x.device, dtype=x.dtype)
        
        # First compute gradient w.r.t. input and accumulate gradients for weight/bias
        # Note: We'll modify grad_output in-place to store dx
        dY_grad_input = grad_output.clone()  # Will be overwritten with dx
        
        # Grid: (B, C)
        grid = (B, C)
        
        # Launch backward kernel (computes dx and accumulates grad_weight, grad_bias)
        if weight is not None:
            instance_norm_backward_kernel[grid](
                dY_grad_input, x, mean, rstd,
                weight,
                grad_bias, grad_weight,
                B, C, H, W,
                stride_b, stride_c, stride_h, stride_w,
                BLOCK_SIZE=256,
            )
        else:
            instance_norm_backward_kernel[grid](
                dY_grad_input, x, mean, rstd,
                None,
                grad_bias, None,
                B, C, H, W,
                stride_b, stride_c, stride_h, stride_w,
                BLOCK_SIZE=256,
            )
        
        # Return gradients for (x, weight, bias, eps)
        return dY_grad_input, grad_weight, grad_bias, None


class InstanceNorm2dTriton(nn.Module):
    """
    Custom Triton-based InstanceNorm2d implementation with affine transformation.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(InstanceNorm2dTriton, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Call autograd function
        return InstanceNormFunction.apply(x, self.weight, self.bias, self.eps)


class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using Triton kernels.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer with Triton kernel.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.inorm = InstanceNorm2dTriton(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.inorm(x)