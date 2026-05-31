import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,     # Input tensor pointer
    weight_ptr,    # Weight tensor pointer
    output_ptr,    # Output tensor pointer
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_depth,
    kernel_height,
    kernel_width,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dilation_d,
    dilation_h,
    dilation_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    
    # Calculate output dimensions
    output_elements = output_depth * output_height * output_width
    
    # Each thread processes one output element
    output_element_idx = tl.program_id(2) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Mask to avoid out-of-bounds access
    mask = output_element_idx < output_elements
    
    # Flatten output coordinates
    out_d = output_element_idx // (output_height * output_width)
    out_h = (output_element_idx % (output_height * output_width)) // output_width
    out_w = output_element_idx % output_width
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for c in range(0, in_channels):
        for kd in range(0, kernel_depth):
            for kh in range(0, kernel_height):
                for kw in range(0, kernel_width):
                    # Compute input position with stride and padding
                    input_d = out_d * stride_d - pad_d + kd * dilation_d
                    input_h = out_h * stride_h - pad_h + kh * dilation_h
                    input_w = out_w * stride_w - pad_w + kw * dilation_w
                    
                    # Check if input position is valid
                    valid_input = (input_d >= 0) & (input_d < input_depth) & \
                                  (input_h >= 0) & (input_h < input_height) & \
                                  (input_w >= 0) & (input_w < input_width)
                    
                    # Load input value if valid
                    input_val = tl.where(valid_input, 
                                        tl.load(input_ptr + 
                                               batch_idx * (in_channels * input_depth * input_height * input_width) +
                                               c * (input_depth * input_height * input_width) +
                                               input_d * (input_height * input_width) +
                                               input_h * input_width +
                                               input_w, 
                                               mask=valid_input, other=0.0),
                                        0.0)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + 
                                        out_channel_idx * (in_channels * kernel_depth * kernel_height * kernel_width) +
                                        c * (kernel_depth * kernel_height * kernel_width) +
                                        kd * (kernel_height * kernel_width) +
                                        kh * kernel_width +
                                        kw)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store result
    output_offset = batch_idx * (out_channels * output_depth * output_height * output_width) + \
                   out_channel_idx * (output_depth * output_height * output_width) + \
                   out_d * (output_height * output_width) + \
                   out_h * output_width + \
                   out_w
    
    tl.store(output_ptr + output_offset, acc, mask=mask)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1)):
    """
    Custom Triton implementation of 3D convolution
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_height = (input_height + 2 * padding[1] - (dilation[1] * (kernel_height - 1) + 1)) // stride[1] + 1
    output_width = (input_width + 2 * padding[2] - (dilation[2] * (kernel_width - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Grid configuration
    grid = (
        batch_size,           # batch dimension
        out_channels,         # output channel dimension  
        math.ceil(output_depth * output_height * output_width / BLOCK_SIZE_M)  # output element dimension
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Ensure proper grouping
        if groups != 1:
            raise ValueError("Groups > 1 not supported in this implementation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

# Helper functions to maintain compatibility with original interface
def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    width = 64
    height = 64
    depth = 64
    x = torch.rand(batch_size, in_channels, width, height, depth)
    return [x]

def get_init_inputs():
    return [3, 64, (3, 5, 7)]