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
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    OUTPUT_BLOCK_SIZE_H: tl.constexpr,
    OUTPUT_BLOCK_SIZE_W: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    
    # Shared memory for input tile
    tile_size = BLOCK_SIZE * BLOCK_SIZE
    shared_input = tl.shared_ptr(input_ptr, tile_size, dtype=tl.float32)
    
    # Calculate output position
    output_h_start = tl.program_id(2) * OUTPUT_BLOCK_SIZE_H
    output_w_start = tl.program_id(3) * OUTPUT_BLOCK_SIZE_W
    
    # Loop over output spatial dimensions
    for out_h in range(output_h_start, min(output_h_start + OUTPUT_BLOCK_SIZE_H, output_height)):
        for out_w in range(output_w_start, min(output_w_start + OUTPUT_BLOCK_SIZE_W, output_width)):
            # Initialize accumulator
            acc = tl.zeros((1,), dtype=tl.float32)
            
            # Loop over input channels and kernel elements
            for c in range(in_channels):
                for kh in range(kernel_height):
                    for kw in range(kernel_width):
                        # Calculate input positions with dilation and padding
                        ih = out_h * stride_h - pad_h + kh * dilation_h
                        iw = out_w * stride_w - pad_w + kw * dilation_w
                        
                        # Check bounds
                        if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                            # Load input value
                            input_val = tl.load(input_ptr + batch_idx * in_channels * input_height * input_width +
                                               c * input_height * input_width +
                                               ih * input_width + iw)
                            
                            # Load weight value
                            weight_val = tl.load(weight_ptr + out_ch_idx * in_channels * kernel_height * kernel_width +
                                                c * kernel_height * kernel_width +
                                                kh * kernel_width + kw)
                            
                            acc += input_val * weight_val
            
            # Store result
            if out_h < output_height and out_w < output_width:
                tl.store(output_ptr + batch_idx * out_channels * output_height * output_width +
                        out_ch_idx * output_height * output_width +
                        out_h * output_width + out_w, acc)

def triton_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Custom Triton implementation of 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 16
    GROUP_SIZE = 4
    OUTPUT_BLOCK_SIZE_H = 8
    OUTPUT_BLOCK_SIZE_W = 8
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        (output_height + OUTPUT_BLOCK_SIZE_H - 1) // OUTPUT_BLOCK_SIZE_H,
        (output_width + OUTPUT_BLOCK_SIZE_W - 1) // OUTPUT_BLOCK_SIZE_W
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
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE,
        OUTPUT_BLOCK_SIZE_H=OUTPUT_BLOCK_SIZE_H,
        OUTPUT_BLOCK_SIZE_W=OUTPUT_BLOCK_SIZE_W
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.
    Optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )

# Test code
batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512
stride = 1
padding = (2, 4)
dilation = (2, 3)

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]