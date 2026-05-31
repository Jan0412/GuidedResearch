import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    groups,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    
    # Calculate output dimensions
    grid_h = (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_ptr(input_ptr, (BLOCK_SIZE_H + 2*padding_h, BLOCK_SIZE_W + 2*padding_w, in_channels))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, out_channels), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific pointers
        group_in_ch = in_channels // groups
        group_out_ch = out_channels // groups
        
        # Get output channel range for this group
        out_ch_start = g * group_out_ch
        out_ch_end = (g + 1) * group_out_ch
        
        # Loop over kernel positions
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input coordinates
                ih_start = out_h_idx * stride_h + kh - padding_h
                iw_start = out_w_idx * stride_w + kw - padding_w
                
                # Load input data
                input_data = tl.load(input_ptr + 
                                   batch_idx * (input_height * input_width * in_channels) +
                                   ih_start * (input_width * in_channels) +
                                   iw_start * in_channels +
                                   tl.arange(0, in_channels)[None, None, :],
                                   mask=(ih_start >= 0) & (ih_start < input_height) &
                                        (iw_start >= 0) & (iw_start < input_width),
                                   other=0.0)
                
                # Load weights for this kernel position
                weight_data = tl.load(weight_ptr +
                                    g * (group_out_ch * group_in_ch * kernel_height * kernel_width) +
                                    tl.arange(0, group_out_ch)[:, None, None] *
                                    (group_in_ch * kernel_height * kernel_width) +
                                    kh * (group_in_ch * kernel_width) +
                                    kw * group_in_ch +
                                    tl.arange(0, group_in_ch)[None, :, None],
                                    other=0.0)
                
                # Perform computation
                acc += tl.expand_dims(input_data, axis=0) * tl.expand_dims(weight_data, axis=1)
    
    # Store output
    tl.store(output_ptr + 
             batch_idx * (output_height * output_width * out_channels) +
             out_h_idx * (output_width * out_channels) +
             out_w_idx * out_channels +
             tl.arange(0, out_channels),
             acc,
             mask=(out_h_idx < output_height) & (out_w_idx < output_width))

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1,1), padding=(0,0), output_padding=(0,0), groups=1):
    """
    Triton implementation of ConvTranspose2d operation
    """
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0]
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1]
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    
    # Grid dimensions
    grid = (
        batch_size,
        (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        groups,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
        
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.
    Optimized using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.bias = bias
        
        # Initialize weight and bias tensors
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Call Triton-based convolution
        return triton_conv_transpose2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )

# Helper functions for compatibility with original API
def get_inputs():
    batch_size = 8
    in_channels = 64
    out_channels = 64
    kernel_size = (3, 7)  # larger asymmetric kernel
    width = 512
    height = 512
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [64, 64, (3, 7)]