import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor pointer (N, C, H, W)
    w_ptr,  # Depthwise kernel pointer (C, 1, KH, KW)
    out_ptr,  # Output tensor pointer (N, C, H_out, W_out)
    n: tl.constexpr,  # Batch size
    c: tl.constexpr,  # Number of channels
    h: tl.constexpr,  # Input height
    w_in: tl.constexpr,  # Input width
    h_out: tl.constexpr,  # Output height
    w_out: tl.constexpr,  # Output width
    kh: tl.constexpr,  # Kernel height
    kw: tl.constexpr,  # Kernel width
    stride: tl.constexpr,  # Stride
    padding: tl.constexpr,  # Padding
    dilation: tl.constexpr,  # Dilation
    BLOCK_SIZE_H: tl.constexpr = 16,
    BLOCK_SIZE_W: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 1,
):
    # Program IDs for output tensor indices
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Calculate input and output positions
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for output indices
    h_mask = out_h < h_out
    w_mask = out_w < w_out
    mask = h_mask[:, None] & w_mask[None, :]
    
    # Compute output position in the input tensor
    in_h = out_h * stride - padding + dilation * tl.arange(0, BLOCK_SIZE_H)[:, None]
    in_w = out_w * stride - padding + dilation * tl.arange(0, BLOCK_SIZE_W)[None, :]
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over kernel spatial dimensions
    for kh_idx in range(kh):
        for kw_idx in range(kw):
            # Compute input position
            in_h_k = in_h + kh_idx * dilation
            in_w_k = in_w + kw_idx * dilation
            
            # Check bounds for input indices
            h_valid = (in_h_k >= 0) & (in_h_k < h)
            w_valid = (in_w_k >= 0) & (in_w_k < w_in)
            valid_mask = h_valid & w_valid
            
            # Compute linear offsets for input
            # We'll handle each channel separately in a loop since BLOCK_SIZE_C=1
            for ch in range(c):
                # Compute input pointer offset for this channel and batch
                offset = pid_n * (c * h * w_in) + ch * (h * w_in)
                
                # Load input values with bounds checking
                input_ptr = x_ptr + offset + in_h_k * w_in + in_w_k
                input_val = tl.load(input_ptr, mask=valid_mask, other=0.0)
                
                # Load kernel value (same for all channels in depthwise conv)
                kernel_offset = ch * (kh * kw) + kh_idx * kw + kw_idx
                kernel_val = tl.load(w_ptr + kernel_offset)
                
                # Accumulate
                acc += input_val * kernel_val
    
    # Store the result
    out_offset = pid_n * (c * h_out * w_out) + tl.arange(0, BLOCK_SIZE_H)[:, None] * (w_out) + tl.arange(0, BLOCK_SIZE_W)[None, :]
    tl.store(out_ptr + out_offset, acc, mask=mask)


@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, H, W)
    w_ptr,  # Pointwise kernel pointer (C_out, C_in, 1, 1)
    out_ptr,  # Output tensor pointer (N, C_out, H, W)
    n: tl.constexpr,  # Batch size
    c_in: tl.constexpr,  # Input channels
    c_out: tl.constexpr,  # Output channels
    h: tl.constexpr,  # Height
    w: tl.constexpr,  # Width
    BLOCK_SIZE_H: tl.constexpr = 8,
    BLOCK_SIZE_W: tl.constexpr = 8,
    BLOCK_SIZE_C: tl.constexpr = 8,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c_out = tl.program_id(2)
    
    # Compute output positions
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks
    h_mask = out_h[:, None] < h
    w_mask = out_w[None, :] < w
    mask = h_mask & w_mask
    
    # Initialize accumulator for each output channel
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for c_start in range(0, c_in, BLOCK_SIZE_C):
        c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
        c_mask = c_offsets < c_in
        
        # Load input values: (H, W, BLOCK_SIZE_C)
        # Compute input offset for current batch
        input_offset = pid_n * (c_in * h * w) + c_offsets[None, None, :] * (h * w) + out_h[:, None, None] * w + out_w[None, :, None]
        input_val = tl.load(x_ptr + input_offset, mask=mask[:, :, None] & c_mask[None, None, :], other=0.0)
        
        # Load weights: (BLOCK_SIZE_C, BLOCK_SIZE_OUT) but we process one output channel at a time
        # For the current output channel (pid_c_out), load weights for all input channels in block
        weight_offsets = pid_c_out * c_in + c_offsets
        weight_val = tl.load(w_ptr + weight_offsets, mask=c_mask)
        
        # Compute dot product across channels
        acc += tl.sum(input_val * weight_val[None, None, :], axis=2)
    
    # Store result
    out_offset = pid_n * (c_out * h * w) + pid_c_out * (h * w) + out_h[:, None] * w + out_w[None, :]
    tl.store(out_ptr + out_offset, acc, mask=mask)


class TritonDepthwiseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, stride=1, padding=0, dilation=1):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        
        # Extract shapes
        n, c, h, w_in = x.shape
        kh, kw = weight.shape[2], weight.shape[3]
        
        # Compute output dimensions
        h_out = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
        w_out = (w_in + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
        
        # Create output tensor
        out = torch.empty((n, c, h_out, w_out), dtype=x.dtype, device=x.device)
        
        # Set up kernel parameters
        grid = (n, triton.cdiv(h_out, 16), triton.cdiv(w_out, 16))
        
        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, weight, out,
            n, c, h, w_in, h_out, w_out,
            kh, kw, stride, padding, dilation,
            BLOCK_SIZE_H=16, BLOCK_SIZE_W=16
        )
        
        # Add bias if provided
        if bias is not None:
            bias_view = bias.view(1, c, 1, 1)
            out = out + bias_view
        
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.input_shape = x.shape
        ctx.output_shape = out.shape
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Not implemented for this example - returning None gradients
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Simple implementation: use PyTorch for backward
            grad_input = torch.nn.functional.conv2d(
                grad_output.contiguous(), weight, None,
                ctx.stride, ctx.padding, ctx.dilation,
                groups=x.shape[1]
            )
        
        if ctx.needs_input_grad[1] and bias is None:
            # Compute weight gradient using PyTorch for simplicity
            grad_weight = torch.nn.functional.conv2d(
                x.permute(1, 0, 2, 3), 
                grad_output.permute(1, 0, 2, 3),
                None, 
                groups=x.shape[1], 
                stride=ctx.stride, 
                padding=ctx.padding, 
                dilation=ctx.dilation
            ).permute(1, 0, 2, 3)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None


class TritonPointwiseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        
        # Extract shapes
        n, c_in, h, w = x.shape
        c_out = weight.shape[0]
        
        # Create output tensor
        out = torch.empty((n, c_out, h, w), dtype=x.dtype, device=x.device)
        
        # Set up kernel parameters
        grid = (n, triton.cdiv(h, 8), c_out)
        
        # Launch kernel
        pointwise_conv2d_kernel[grid](
            x, weight, out,
            n, c_in, c_out, h, w,
            BLOCK_SIZE_H=8, BLOCK_SIZE_W=8, BLOCK_SIZE_C=8
        )
        
        # Add bias if provided
        if bias is not None:
            bias_view = bias.view(1, c_out, 1, 1)
            out = out + bias_view
        
        ctx.save_for_backward(x, weight, bias)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Use PyTorch for backward
            grad_input = torch.nn.functional.conv2d(
                grad_output.contiguous(), weight,
                None
            )
        
        if ctx.needs_input_grad[1]:
            # Compute weight gradient
            grad_weight = torch.nn.functional.conv2d(
                x.permute(1, 0, 2, 3), 
                grad_output.permute(1, 0, 2, 3),
                None
            ).permute(1, 0, 2, 3)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias


class TritonDepthwiseConv2dLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return TritonDepthwiseConv2d.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


class TritonPointwiseConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return TritonPointwiseConv2d.apply(x, self.weight, self.bias)


import math


class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation with optimized Triton kernels.
    
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise = TritonDepthwiseConv2dLayer(
            in_channels, kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            bias=bias
        )
        self.pointwise = TritonPointwiseConv2dLayer(
            in_channels, out_channels, 
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x