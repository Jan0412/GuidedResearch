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
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get block IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    output_elem_id = tl.program_id(2)
    
    # Calculate output indices
    output_d = output_elem_id // (output_height * output_width)
    output_h = (output_elem_id % (output_height * output_width)) // output_width
    output_w = output_elem_id % output_width
    
    # Check bounds
    if output_d >= output_depth or output_h >= output_height or output_w >= output_width:
        return
        
    # Calculate input coordinates
    input_d = output_d * stride_d - padding_d
    input_h = output_h * stride_h - padding_h
    input_w = output_w * stride_w - padding_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions and input channels
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                for ic in range(in_channels):
                    # Calculate input coordinates
                    id = input_d + kd
                    ih = input_h + kh
                    iw = input_w + kw
                    
                    # Check if input coordinate is valid
                    if id >= 0 and id < input_depth and ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                        # Calculate input index
                        input_idx = batch_id * (in_channels * input_depth * input_height * input_width) + \
                                   ic * (input_depth * input_height * input_width) + \
                                   id * (input_height * input_width) + \
                                   ih * input_width + \
                                   iw
                        
                        # Calculate weight index
                        weight_idx = channel_id * (in_channels * kernel_depth * kernel_height * kernel_width) + \
                                    ic * (kernel_depth * kernel_height * kernel_width) + \
                                    kd * (kernel_height * kernel_width) + \
                                    kh * kernel_width + \
                                    kw
                        
                        # Load values and accumulate
                        input_val = tl.load(input_ptr + input_idx, mask=True)
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                        acc += input_val * weight_val
    
    # Store result
    output_idx = batch_id * (out_channels * output_depth * output_height * output_width) + \
                 channel_id * (output_depth * output_height * output_width) + \
                 output_d * (output_height * output_width) + \
                 output_h * output_width + \
                 output_w
    
    tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0)):
    """
    Triton implementation of ConvTranspose3d
    """
    # Extract dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    
    output_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_depth
    output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height
    output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous and on correct device
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Launch kernel
    grid = (
        batch_size,  # batch dimension
        out_channels,  # channel dimension
        output_depth * output_height * output_width  # output elements
    )
    
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 1
    
    conv_transpose3d_kernel[grid](
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
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use our Triton kernel implementation
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding)
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"output_padding={self.output_padding}, groups={self.groups}, bias={self.bias is not None}"
        )