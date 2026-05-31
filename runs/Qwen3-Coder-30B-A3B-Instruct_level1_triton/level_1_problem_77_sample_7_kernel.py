import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_shape,
    weight_shape,
    output_shape,
    stride,
    padding,
    dilation,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    output_element_id = tl.program_id(2)
    
    # Calculate global output indices
    output_idx = output_element_id * OUTPUT_ELEMENTS_PER_BLOCK + tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK)
    output_depth_idx = output_idx // (output_height * output_width)
    remaining = output_idx % (output_height * output_width)
    output_height_idx = remaining // output_width
    output_width_idx = remaining % output_width
    
    # Mask for valid output elements
    valid_mask = (output_idx < output_depth * output_height * output_width) & \
                 (output_depth_idx < output_depth) & \
                 (output_height_idx < output_height) & \
                 (output_width_idx < output_width)
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(0, in_channels, CHANNELS_PER_BLOCK):
        # Load weight block
        weight_block = tl.load(weight_ptr + 
                              (channel_id * in_channels + ic) * (kernel_size ** 3) + 
                              tl.arange(0, CHANNELS_PER_BLOCK)[:, None, None] * (kernel_size ** 3) +
                              tl.arange(0, kernel_size)[None, :, None] * (kernel_size ** 2) +
                              tl.arange(0, kernel_size)[None, None, :] * kernel_size,
                              mask=(tl.arange(0, CHANNELS_PER_BLOCK)[:, None, None] < in_channels - ic),
                              other=0.0)
        
        # Calculate input coordinates for this output position
        input_d = output_depth_idx * stride - padding + tl.arange(0, kernel_size)[:, None, None] * dilation
        input_h = output_height_idx * stride - padding + tl.arange(0, kernel_size)[None, :, None] * dilation
        input_w = output_width_idx * stride - padding + tl.arange(0, kernel_size)[None, None, :] * dilation
        
        # Check bounds
        valid_input_mask = (input_d >= 0) & (input_d < input_depth) & \
                          (input_h >= 0) & (input_h < input_height) & \
                          (input_w >= 0) & (input_w < input_width)
        
        # Load input data
        input_data = tl.load(input_ptr + 
                            (batch_id * in_channels + ic) * (input_depth * input_height * input_width) + 
                            input_d * (input_height * input_width) + 
                            input_h * input_width + 
                            input_w,
                            mask=valid_input_mask,
                            other=0.0)
        
        # Compute convolution
        acc += tl.sum(weight_block * input_data, axis=(1, 2, 3))
    
    # Store results
    output_offset = (batch_id * out_channels + channel_id) * (output_depth * output_height * output_width) + output_idx
    tl.store(output_ptr + output_offset, acc, mask=valid_mask)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Triton implementation of 3D transposed convolution
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get shapes
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride - 2 * padding + dilation * (kernel_depth - 1) + 1
    output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_height - 1) + 1
    output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_width - 1) + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define kernel parameters
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 8
    OUTPUT_ELEMENTS_PER_BLOCK = 32
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        out_channels,
        (output_depth * output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_tensor.shape,
        weight.shape,
        output.shape,
        stride,
        padding,
        dilation,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )

# For compatibility with the test code
def get_inputs():
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    depth = 16
    height = 32
    width = 32
    stride = 2
    padding = 1
    dilation = 2
    
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [32, 64, 3, 2, 1, 2]