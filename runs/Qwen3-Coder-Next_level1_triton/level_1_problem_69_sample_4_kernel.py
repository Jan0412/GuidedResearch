import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to tensors
    x_ptr,  # Input tensor: (batch, in_channels, H_in, W_in)
    w_ptr,  # Weight tensor: (in_channels, out_channels, kH, kW)
    b_ptr,  # Bias tensor: (out_channels,) or None
    out_ptr,  # Output tensor: (batch, out_channels, H_out, W_out)
    # Tensor dimensions
    batch_size, in_channels, out_channels,
    height_in, width_in,
    height_out, width_out,
    kernel_height, kernel_width,
    stride_h, stride_w,
    padding_h, padding_w,
    output_padding_h, output_padding_w,
    dilation_h, dilation_w,
    # Strides for memory access
    stride_x_bs, stride_x_ic, stride_x_h, stride_x_w,
    stride_w_ic, stride_w_oc, stride_w_kh, stride_w_kw,
    stride_b_oc,
    stride_out_bs, stride_out_oc, stride_out_h, stride_out_w,
    # Block sizes
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_oc = tl.program_id(1)  # output channel index
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index
    
    # Calculate output position
    out_h = pid_h
    out_w = pid_w
    
    # Initialize accumulator for this output position
    acc = tl.zeros((BLOCK_SIZE_OC,), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for ic in range(0, in_channels, BLOCK_SIZE_IC):
        for kh in range(0, kernel_height, BLOCK_SIZE_KH):
            for kw in range(0, kernel_width, BLOCK_SIZE_KW):
                # Calculate input position from output position
                # For transposed convolution: in_h = (out_h - (kH - 1) * dilation_h - 1 + padding_h) // stride_h + 1
                # But more directly: out_h = in_h * stride_h - padding_h + (kH - 1) * dilation_h + output_padding_h
                # => in_h = (out_h + padding_h - (kH - 1) * dilation_h - output_padding_h) // stride_h
                
                # Check if this input position is valid
                in_h_start = (out_h + padding_h - (kh + 0) * dilation_h - output_padding_h) // stride_h
                in_w_start = (out_w + padding_w - (kw + 0) * dilation_w - output_padding_w) // stride_w
                
                # Check bounds for this kernel position
                valid_h = (in_h_start >= 0) & (in_h_start < height_in)
                valid_w = (in_w_start >= 0) & (in_w_start < width_in)
                
                if valid_h and valid_w:
                    # Load input value
                    x_offset = pid_b * stride_x_bs + (ic + tl.arange(0, BLOCK_SIZE_IC)) * stride_x_ic + in_h_start * stride_x_h + in_w_start * stride_x_w
                    x_mask = (ic + tl.arange(0, BLOCK_SIZE_IC)) < in_channels
                    x_val = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
                    
                    # Load weights for this kernel position
                    # Weight shape: (in_channels, out_channels, kH, kW)
                    w_offset = (ic + tl.arange(0, BLOCK_SIZE_IC))[:, None] * stride_w_ic + \
                               (pid_oc * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC))[None, :] * stride_w_oc + \
                               kh * stride_w_kh + kw * stride_w_kw
                    w_mask = ((ic + tl.arange(0, BLOCK_SIZE_IC))[:, None] < in_channels) & \
                             ((pid_oc * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC))[None, :] < out_channels)
                    w_val = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)
                    
                    # Accumulate: output[oc] += input[ic] * weight[ic, oc, kh, kw]
                    acc += tl.sum(x_val[:, None] * w_val, axis=0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_oc * stride_b_oc)
        acc += bias
    
    # Store result
    out_offset = pid_b * stride_out_bs + (pid_oc * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)) * stride_out_oc + \
                 out_h * stride_out_h + out_w * stride_out_w
    out_mask = (pid_oc * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)) < out_channels
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=out_mask)


def triton_conv_transpose2d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    """
    Triton implementation of ConvTranspose2d
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height_in, width_in = x.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    output_padding_h, output_padding_w = output_padding
    dilation_h, dilation_w = dilation
    
    height_out = (height_in - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_height - 1) + output_padding_h + 1
    width_out = (width_in - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + output_padding_w + 1
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, height_out, width_out, dtype=x.dtype, device=x.device)
    
    # Configure block sizes - choose based on problem size
    BLOCK_SIZE_OC = min(32, out_channels)  # Output channel block size
    BLOCK_SIZE_IC = min(16, in_channels)   # Input channel block size
    BLOCK_SIZE_KH = min(4, kernel_height)  # Kernel height block size
    BLOCK_SIZE_KW = min(4, kernel_width)   # Kernel width block size
    
    # Grid dimensions: (batch, output_channels, output_height, output_width)
    # We'll use a simpler grid with separate kernel for better performance
    grid = (batch_size, out_channels, height_out, width_out)
    
    # Strides
    stride_x_bs = x.stride(0)
    stride_x_ic = x.stride(1)
    stride_x_h = x.stride(2)
    stride_x_w = x.stride(3)
    
    stride_w_ic = weight.stride(0)
    stride_w_oc = weight.stride(1)
    stride_w_kh = weight.stride(2)
    stride_w_kw = weight.stride(3)
    
    stride_b_oc = bias.stride(0) if bias is not None else 0
    
    stride_out_bs = out.stride(0)
    stride_out_oc = out.stride(1)
    stride_out_h = out.stride(2)
    stride_out_w = out.stride(3)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        height_in, width_in,
        height_out, width_out,
        kernel_height, kernel_width,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w,
        dilation_h, dilation_w,
        stride_x_bs, stride_x_ic, stride_x_h, stride_x_w,
        stride_w_ic, stride_w_oc, stride_w_kh, stride_w_kw,
        stride_b_oc,
        stride_out_bs, stride_out_oc, stride_out_h, stride_out_w,
        BLOCK_SIZE_IC=BLOCK_SIZE_IC,
        BLOCK_SIZE_OC=BLOCK_SIZE_OC,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), output_padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize the same way as original
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters manually
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 2D convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding,
            self.dilation, self.groups
        )