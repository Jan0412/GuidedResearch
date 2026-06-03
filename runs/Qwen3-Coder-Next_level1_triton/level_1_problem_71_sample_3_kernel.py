import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Tuple, Optional

@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, h_in, w_in)
    w_ptr,  # Weight tensor: (in_channels, out_channels, k_h, k_w)
    b_ptr,  # Bias tensor: (out_channels,) or None
    y_ptr,  # Output tensor: (batch, out_channels, h_out, w_out)
    batch_size, in_channels, out_channels,
    h_in, w_in, h_out, w_out,
    k_h, k_w,
    stride_h, stride_w,
    pad_h, pad_w,
    output_pad_h, output_pad_w,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for accumulation (in_channels)
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    
    # Output spatial positions
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Compute input positions that contribute to this output
    # For transposed convolution: h_in = (h_out - 1 - output_pad_h - 2*pad_h + k_h) // stride_h + 1
    h_offset = pid_h * stride_h - pad_h
    w_offset = pid_w * stride_w - pad_w
    
    # Accumulator for output
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels
    for k in range(0, in_channels, BLOCK_SIZE_K):
        k_range = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_range < in_channels
        
        # Load input: x[pid_batch, k, h_in_pos, w_in_pos]
        # Need to compute h_in_pos and w_in_pos such that:
        # pid_h = h_in_pos * stride_h + k_h - 1 - pad_h + output_pad_h (conceptually)
        # Instead, for each k, we check if the input position is valid
        
        # For each output position (pid_h, pid_w), we look at input positions
        # h_in_pos = pid_h * stride_h + k_h - 1 - pad_h - k_h_offset (for k_h_offset in [0, k_h-1])
        # Actually, transposed conv: output[h_out, w_out] = sum_{k_h, k_w} input[h_in, w_in] * weight[in_c, out_c, k_h, k_w]
        # where h_in = (h_out - k_h + 1 + 2*pad_h - output_pad_h) // stride_h + k_h_offset
        # This is complex; better to iterate over kernel positions
        
    # Alternative approach: iterate over kernel positions
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Iterate over kernel positions
    for kh in range(k_h):
        for kw in range(k_w):
            # Compute corresponding input position
            h_in_pos = pid_h * stride_h + kh - pad_h
            w_in_pos = pid_w * stride_w + kw - pad_w
            
            # Check if input position is valid
            if h_in_pos >= 0 and h_in_pos < h_in and w_in_pos >= 0 and w_in_pos < w_in:
                # Load input: x[pid_batch, :, h_in_pos, w_in_pos]
                x_offset = pid_batch * (in_channels * h_in * w_in) + \
                          tl.arange(0, BLOCK_SIZE_K) * (h_in * w_in) + \
                          h_in_pos * w_in + w_in_pos
                x_mask = tl.arange(0, BLOCK_SIZE_K) < in_channels
                x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
                
                # Load weight: weight[in_c, pid_out_c, kh, kw]
                w_offset = tl.arange(0, BLOCK_SIZE_K) * (out_channels * k_h * k_w) + \
                          pid_out_c * (k_h * k_w) + kh * k_w + kw
                w_mask = tl.arange(0, BLOCK_SIZE_K) < in_channels
                w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)
                
                # Accumulate: acc += x_vals * w_vals
                acc += x_vals * w_vals
    
    # Apply bias if present
    if b_ptr is not None:
        bias_offset = pid_out_c
        bias_val = tl.load(b_ptr + bias_offset)
        acc += bias_val
    
    # Store result
    y_offset = pid_batch * (out_channels * h_out * w_out) + \
              pid_out_c * (h_out * w_out) + \
              pid_h * w_out + pid_w
    tl.store(y_ptr + y_offset, acc.to(x_ptr.dtype.element_ty))


class TritonConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, 
                stride: Tuple[int, int], padding: Tuple[int, int], 
                output_padding: Tuple[int, int], groups: int):
        
        # Extract parameters
        batch_size, in_channels, h_in, w_in = x.shape
        out_channels, _, k_h, k_w = weight.shape
        
        # Calculate output dimensions
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        output_pad_h, output_pad_w = output_padding
        
        h_out = (h_in - 1) * stride_h - 2 * pad_h + k_h + output_pad_h
        w_out = (w_in - 1) * stride_w - 2 * pad_w + k_w + output_pad_w
        
        # Prepare output tensor
        y = torch.empty((batch_size, out_channels, h_out, w_out), 
                       dtype=x.dtype, device=x.device)
        
        # Kernel configuration
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 1
        BLOCK_SIZE_K = 32
        
        # Grid dimensions
        grid = lambda meta: (
            batch_size,
            out_channels,
            h_out,
            w_out
        )
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, weight, bias, y,
            batch_size, in_channels, out_channels,
            h_in, w_in, h_out, w_out,
            k_h, k_w,
            stride_h, stride_w,
            pad_h, pad_w,
            output_pad_h, output_pad_w,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        # For simplicity, fall back to PyTorch implementation for backward
        x, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        groups = ctx.groups
        
        # Use PyTorch's native backward for gradients
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(x.shape, weight, grad_output, 
                                                   stride=stride, padding=padding,
                                                   output_padding=output_padding,
                                                   groups=groups)
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(x, weight.shape, grad_output,
                                                     stride=stride, padding=padding,
                                                     groups=groups)
        
        if ctx.needs_input_grad[2] and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None, None


def triton_conv_transpose2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                           stride=1, padding=0, output_padding=0, groups=1):
    # Convert parameters to tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    
    return TritonConvTranspose2d.apply(input, weight, bias, 
                                      stride, padding, output_padding, groups)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.
    Optimized with custom Triton kernel for forward pass.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Register buffers for parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel for forward pass.
        """
        return triton_conv_transpose2d(x, self.weight, self.bias,
                                      self.stride, self.padding, 
                                      self.output_padding, self.groups)