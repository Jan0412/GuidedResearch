import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,     # Input tensor pointer
    weight_ptr,    # Weight tensor pointer
    output_ptr,    # Output tensor pointer
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_row = tl.program_id(2)
    
    # Calculate output dimensions
    num_output_rows = output_height
    num_output_cols = output_width
    
    # Shared memory for input tile
    TILE_M = BLOCK_SIZE_M
    TILE_N = BLOCK_SIZE_N
    TILE_K = BLOCK_SIZE_K
    
    # Initialize accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    
    # Loop over kernel elements
    for kh in range(0, kernel_height):
        for kw in range(0, kernel_width):
            # Calculate input indices
            input_row = out_row * stride_h + kh * dilation_h - pad_h
            input_col = 0
            
            # Load input tile
            input_tile = tl.load(input_ptr + 
                               batch_idx * (in_channels * input_height * input_width) +
                               tl.arange(0, TILE_M)[:, None] * (input_width * in_channels) +
                               input_row * input_width * in_channels +
                               tl.arange(0, TILE_N)[None, :] * in_channels +
                               tl.arange(0, TILE_K)[None, :] * in_channels + 
                               kw * dilation_w, 
                               mask=(tl.arange(0, TILE_M)[:, None] < num_output_rows) &
                                    (tl.arange(0, TILE_N)[None, :] < num_output_cols) &
                                    (input_row >= 0) & (input_row < input_height) &
                                    (tl.arange(0, TILE_K)[None, :] < in_channels), 
                               other=0.0)
            
            # Load weight tile
            weight_tile = tl.load(weight_ptr + 
                                out_channel_idx * (in_channels * kernel_height * kernel_width) +
                                tl.arange(0, TILE_K)[None, :] * (kernel_height * kernel_width) +
                                kh * kernel_width + kw, 
                                mask=tl.arange(0, TILE_K)[None, :] < in_channels,
                                other=0.0)
            
            # Accumulate
            acc += tl.dot(input_tile, weight_tile)
    
    # Store output
    tl.store(output_ptr + 
             batch_idx * (out_channels * output_height * output_width) +
             out_channel_idx * (output_height * output_width) +
             out_row * output_width + 
             tl.arange(0, TILE_N),
             acc, 
             mask=tl.arange(0, TILE_N) < num_output_cols)

def triton_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Grid configuration
    grid = (
        batch_size,           # batch dimension
        out_channels,         # output channel dimension  
        (output_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M  # output row dimension
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)