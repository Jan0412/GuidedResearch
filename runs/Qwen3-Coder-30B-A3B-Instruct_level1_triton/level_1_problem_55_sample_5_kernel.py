import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    batch_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    out_c_idx = tl.program_id(3)
    
    # Shared memory for input tile
    shared_input = tl.shared_tile(input_ptr, (BLOCK_SIZE_H + 2 * padding_h, BLOCK_SIZE_W + 2 * padding_w))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over tiles within the kernel
    for kh in range(0, kernel_h, TILE_SIZE):
        for kw in range(0, kernel_w, TILE_SIZE):
            # Compute current tile bounds
            tile_h_start = kh
            tile_w_start = kw
            
            # Load weight tile
            w_offset = out_c_idx * in_channels * kernel_h * kernel_w + \
                      tile_h_start * in_channels * kernel_w + \
                      tile_w_start * in_channels
            
            # Load input tile
            input_h_start = out_h_idx * stride_h - padding_h
            input_w_start = out_w_idx * stride_w - padding_w
            
            # Load input region with padding handling
            for ih in range(BLOCK_SIZE_H):
                for iw in range(BLOCK_SIZE_W):
                    h = input_h_start + ih
                    w = input_w_start + iw
                    
                    # Check bounds
                    if h >= 0 and h < input_height and w >= 0 and w < input_width:
                        input_offset = batch_idx * in_channels * input_height * input_width + \
                                     ih * input_width + iw
                        shared_input[ih, iw] = tl.load(input_ptr + input_offset)
                    else:
                        shared_input[ih, iw] = 0.0
                    
            # Compute convolution for this tile
            for ih in range(min(TILE_SIZE, kernel_h - kh)):
                for iw in range(min(TILE_SIZE, kernel_w - kw)):
                    if kh + ih < kernel_h and kw + iw < kernel_w:
                        # Load weight
                        w_idx = out_c_idx * in_channels * kernel_h * kernel_w + \
                               (kh + ih) * in_channels * kernel_w + \
                               (kw + iw) * in_channels
                        
                        # Load input
                        input_idx = ih * input_width + iw
                        input_val = shared_input[ih, iw]
                        
                        # Accumulate
                        acc += input_val * tl.load(weight_ptr + w_idx)
    
    # Add bias if present
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + out_c_idx)
    
    # Store result
    output_offset = batch_idx * out_channels * output_height * output_width + \
                   out_c_idx * output_height * output_width + \
                   out_h_idx * output_width + out_w_idx
    
    tl.store(output_ptr + output_offset, acc)


def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    output_width = (input_width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 8
    TILE_SIZE = 8
    
    # Create grid
    grid = (
        batch_size,
        output_height,
        output_width,
        out_channels
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        batch_size,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        TILE_SIZE=TILE_SIZE
    )
    
    return output


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.
    Optimized using custom Triton kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation)
        )