import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for transposed 2D convolution forward pass
@triton.jit
def conv_transpose2d_kernel(
    # Input tensor (batch, in_channels, h, w)
    x_ptr,
    # Weight tensor (in_channels, out_channels, k_h, k_w)
    w_ptr,
    # Optional bias tensor (out_channels,)
    b_ptr,
    # Output tensor (batch, out_channels, o_h, o_w)
    out_ptr,
    # Dimensions
    batch_size, in_channels, out_channels,
    height_in, width_in,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    output_padding_h, output_padding_w,
    dilation_h, dilation_w,
    out_height, out_width,
    # Block sizes for tiling
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)  # batch
    pid_oh = tl.program_id(1)  # output height block
    pid_ow = tl.program_id(2)  # output width block
    
    # Calculate output spatial indices
    out_h_start = pid_oh * BLOCK_H
    out_w_start = pid_ow * BLOCK_W
    
    # Output indices range
    out_h_offsets = out_h_start + tl.arange(0, BLOCK_H)
    out_w_offsets = out_w_start + tl.arange(0, BLOCK_W)
    
    # Create masks for valid indices
    out_h_mask = out_h_offsets < out_height
    out_w_mask = out_w_offsets < out_width
    out_hw_mask = out_h_mask[:, None] & out_w_mask[None, :]
    
    # Initialize accumulator for output
    output = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate corresponding input positions
            in_h = (out_h_start + tl.arange(0, BLOCK_H) - kh * dilation_h + padding_h) // stride_h
            in_w = (out_w_start + tl.arange(0, BLOCK_W) - kw * dilation_w + padding_w) // stride_w
            
            # Check if the input indices are valid
            valid_h = ((out_h_start + tl.arange(0, BLOCK_H) - kh * dilation_h + padding_h) % stride_h == 0)
            valid_w = ((out_w_start + tl.arange(0, BLOCK_W) - kw * dilation_w + padding_w) % stride_w == 0)
            valid = valid_h[:, None] & valid_w[None, :]
            
            in_h = tl.maximum(0, tl.minimum(in_h, height_in - 1))
            in_w = tl.maximum(0, tl.minimum(in_w, width_in - 1))
            
            # Calculate input tensor offsets
            x_offsets = (
                pid_b * (in_channels * height_in * width_in) +
                tl.arange(0, in_channels)[:, None, None] * (height_in * width_in) +
                in_h[None, :, :] * width_in +
                in_w[None, :, :]
            )
            
            # Calculate weight tensor offsets
            w_offsets = (
                tl.arange(0, in_channels)[:, None] * (out_channels * kernel_h * kernel_w) +
                kh * (out_channels * kernel_w) +
                kw * out_channels +
                tl.arange(0, out_channels)[None, :]
            )
            
            # Load data
            x = tl.load(x_ptr + x_offsets, mask=valid[None, :, :], other=0.0)
            w = tl.load(w_ptr + w_offsets, mask=tl.arange(0, in_channels)[:, None] < in_channels, other=0.0)
            
            # Accumulate: x * w
            # x has shape (in_channels, BLOCK_H, BLOCK_W)
            # w has shape (in_channels, out_channels)
            # Result should be (BLOCK_H, BLOCK_W, out_channels) -> (BLOCK_H, BLOCK_W)
            output += tl.sum(x * w[:, :, None], axis=0)
    
    # Add bias if available
    if b_ptr is not None:
        b = tl.load(b_ptr + tl.arange(0, out_channels), mask=tl.arange(0, out_channels) < out_channels)
        output += b[None, :]
    
    # Store output
    out_offsets = (
        pid_b * (out_channels * out_height * out_width) +
        tl.arange(0, out_channels)[:, None, None] * (out_height * out_width) +
        out_h_offsets[None, :, None] * out_width +
        out_w_offsets[None, None, :]
    )
    
    # Transpose to get final shape (out_channels, BLOCK_H, BLOCK_W)
    output = tl.trans(output[None, :, :])
    
    tl.store(out_ptr + out_offsets, output, mask=tl.arange(0, out_channels)[:, None, None] < out_channels & out_hw_mask[None, :, :])


def triton_conv_transpose2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), dilation=(1, 1)):
    """
    Triton-based transposed 2D convolution implementation.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height_in, width_in = x.shape
    out_channels, in_channels_w, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions (same as PyTorch)
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    output_padding_h, output_padding_w = output_padding
    dilation_h, dilation_w = dilation
    
    out_height = (height_in - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1
    out_width = (width_in - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1
    
    # Allocate output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Configure block sizes
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_K = 16
    
    # Grid dimensions: (batch, output_height_blocks, output_width_blocks)
    grid = (
        batch_size,
        (out_height + BLOCK_H - 1) // BLOCK_H,
        (out_width + BLOCK_W - 1) // BLOCK_W,
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        height_in, width_in,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w,
        dilation_h, dilation_w,
        out_height, out_width,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_K=BLOCK_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the transposed 2D convolution model using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using our Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, dilation=self.dilation
        )


import math