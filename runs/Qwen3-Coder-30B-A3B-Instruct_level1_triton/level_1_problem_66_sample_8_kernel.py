import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
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
    padding_d,
    padding_h,
    padding_w,
    dilation_d,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d = tl.program_id(2)
    
    # Shared memory for input tile
    tile_size = BLOCK_SIZE * GROUP_SIZE
    input_tile = tl.shared.tensor([tile_size], tl.float32)
    
    # Calculate output dimensions
    if out_d >= output_depth:
        return
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels and kernel spatial dimensions
    for ic in range(0, in_channels, GROUP_SIZE):
        # Load weights for this channel and kernel position
        weight_offset = out_ch_idx * in_channels * kernel_depth * kernel_height * kernel_width + \
                       ic * kernel_depth * kernel_height * kernel_width
        
        # Process kernel spatial dimensions
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input positions
                    input_d = out_d * stride_d - padding_d + kd * dilation_d
                    input_h = out_d * stride_h - padding_h + kh * dilation_h
                    input_w = out_d * stride_w - padding_w + kw * dilation_w
                    
                    # Check bounds
                    if (input_d >= 0 and input_d < input_depth and 
                        input_h >= 0 and input_h < input_height and 
                        input_w >= 0 and input_w < input_width):
                        
                        # Load input data
                        input_offset = batch_idx * in_channels * input_depth * input_height * input_width + \
                                     ic * input_depth * input_height * input_width + \
                                     input_d * input_height * input_width + \
                                     input_h * input_width + \
                                     input_w
                        
                        # Load weight
                        weight_val = tl.load(weight_ptr + weight_offset + kd * kernel_height * kernel_width + \
                                           kh * kernel_width + kw)
                        
                        # Load input value
                        input_val = tl.load(input_ptr + input_offset)
                        
                        # Accumulate
                        acc += weight_val * input_val
    
    # Store result
    output_offset = batch_idx * out_channels * output_depth * output_height * output_width + \
                   out_ch_idx * output_depth * output_height * output_width + \
                   out_d * output_height * output_width + \
                   out_d * output_width + \
                   out_d
    
    tl.store(output_ptr + output_offset, acc)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    """
    Triton implementation of 3D convolution
    """
    # Extract parameters
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_height = (input_height + 2 * padding[1] - (dilation[1] * (kernel_height - 1) + 1)) // stride[1] + 1
    output_width = (input_width + 2 * padding[2] - (dilation[2] * (kernel_width - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define grid configuration
    grid = (
        batch_size,  # batch dimension
        out_channels,  # output channels
        output_depth  # output depth
    )
    
    # Launch kernel
    BLOCK_SIZE = 32
    GROUP_SIZE = 8
    
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
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Set requires_grad to True for training
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Use our Triton-based convolution implementation
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )

# Helper function to create proper tensor shapes for testing
def get_inputs():
    batch_size = 8
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    depth = 16
    height = 128
    width = 128
    x = torch.rand(batch_size, in_channels, depth, height, width, dtype=torch.float32)
    return [x]

def get_init_inputs():
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    return [in_channels, out_channels, kernel_size]