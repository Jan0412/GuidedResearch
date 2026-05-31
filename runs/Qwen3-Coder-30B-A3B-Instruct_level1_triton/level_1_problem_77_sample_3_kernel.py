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
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_elem_idx = tl.program_id(2)
    
    # Calculate which output element this block processes
    output_offset = output_elem_idx * OUTPUT_ELEMENTS_PER_BLOCK
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Process multiple channels per block if needed
    for c in range(channel_idx * CHANNELS_PER_BLOCK, min((channel_idx + 1) * CHANNELS_PER_BLOCK, out_channels)):
        # Initialize accumulator
        acc = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
        
        # For each kernel position
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input coordinates
                    input_d = (output_elem_idx * stride_d + kd * dilation_d - padding_d)
                    input_h = (output_elem_idx * stride_h + kh * dilation_h - padding_h)
                    input_w = (output_elem_idx * stride_w + kw * dilation_w - padding_w)
                    
                    # Check bounds
                    if (input_d >= 0 and input_d < input_depth and
                        input_h >= 0 and input_h < input_height and
                        input_w >= 0 and input_w < input_width):
                        
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          batch_idx * (in_channels * input_depth * input_height * input_width) +
                                          c * (input_depth * input_height * input_width) +
                                          input_d * (input_height * input_width) +
                                          input_h * input_width +
                                          input_w)
                        
                        # Load weight
                        weight_val = tl.load(weight_ptr + 
                                           c * (out_channels * kernel_depth * kernel_height * kernel_width) +
                                           (kd * kernel_height * kernel_width + kh * kernel_width + kw))
                        
                        # Accumulate
                        acc += input_val * weight_val
        
        # Store output
        tl.store(output_ptr + 
                batch_idx * (out_channels * output_depth * output_height * output_width) +
                c * (output_depth * output_height * output_width) +
                output_elem_idx, acc)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1)):
    """
    Triton implementation of 3D transposed convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_depth - 1) + 1
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_height - 1) + 1
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_width - 1) + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Flatten output for processing
    output_flat = output.view(batch_size, out_channels, -1)
    
    # Grid configuration
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 4
    OUTPUT_ELEMENTS_PER_BLOCK = 32
    
    grid = (
        batch_size,
        (out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,
        (output_depth * output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output_flat,
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
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
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