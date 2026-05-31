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
    input_width,
    input_height,
    output_depth,
    output_width,
    output_height,
    kernel_depth,
    kernel_width,
    kernel_height,
    stride_d,
    stride_w,
    stride_h,
    padding_d,
    padding_w,
    padding_h,
    dilation_d,
    dilation_w,
    dilation_h,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate output coordinates
    output_d = output_idx // (output_width * output_height)
    output_w = (output_idx % (output_width * output_height)) // output_height
    output_h = output_idx % output_height
    
    # Check bounds
    if output_d >= output_depth or output_w >= output_width or output_h >= output_height:
        return
        
    # Shared memory for input tile
    input_tile = tl.shared_ptr(tl.float32, shape=(CHANNELS_PER_BLOCK, kernel_depth, kernel_width, kernel_height))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(0, in_channels, CHANNELS_PER_BLOCK):
        # Load weights for this channel group
        weight_block = tl.load(weight_ptr + 
                              (channel_idx * in_channels + c) * kernel_depth * kernel_width * kernel_height +
                              tl.arange(0, kernel_depth)[:, None, None] * kernel_width * kernel_height +
                              tl.arange(0, kernel_width)[None, :, None] * kernel_height +
                              tl.arange(0, kernel_height)[None, None, :])
        
        # Load input region for this channel group
        for kd in range(kernel_depth):
            for kw in range(kernel_width):
                for kh in range(kernel_height):
                    input_d = output_d * stride_d - padding_d + kd * dilation_d
                    input_w = output_w * stride_w - padding_w + kw * dilation_w
                    input_h = output_h * stride_h - padding_h + kh * dilation_h
                    
                    # Check if input position is valid
                    if (input_d >= 0 and input_d < input_depth and
                        input_w >= 0 and input_w < input_width and
                        input_h >= 0 and input_h < input_height):
                        
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          (batch_idx * in_channels + c) * input_depth * input_width * input_height +
                                          input_d * input_width * input_height +
                                          input_w * input_height +
                                          input_h)
                        
                        # Accumulate
                        acc += input_val * weight_block[kd, kw, kh]
    
    # Store result
    tl.store(output_ptr + 
             (batch_idx * out_channels + channel_idx) * output_depth * output_width * output_height +
             output_d * output_width * output_height +
             output_w * output_height +
             output_h, acc)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1)):
    """
    Triton implementation of 3D convolution
    """
    batch_size, in_channels, input_depth, input_width, input_height = input_tensor.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    output_height = (input_height + 2 * padding[2] - (dilation[2] * (kernel_height - 1) + 1)) // stride[2] + 1
    
    # Allocate output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_width, output_height, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous and on GPU
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define kernel launch parameters
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 16
    OUTPUT_ELEMENTS_PER_BLOCK = 32
    
    # Grid dimensions
    grid = (
        batch_size,  # batch dimension
        out_channels,  # output channels
        output_depth * output_width * output_height  # output elements
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
        input_width,
        input_height,
        output_depth,
        output_width,
        output_height,
        kernel_depth,
        kernel_width,
        kernel_height,
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
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias, 
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            dilation=(self.dilation, self.dilation, self.dilation)
        )