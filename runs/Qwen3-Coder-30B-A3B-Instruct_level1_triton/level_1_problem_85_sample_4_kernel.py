import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate global output position
    output_row = output_idx // output_width
    output_col = output_idx % output_width
    
    # Shared memory for input tile and kernel
    input_tile = tl.shared.tensor([CHANNELS_PER_BLOCK, kernel_height, kernel_width], tl.float32)
    weight_tile = tl.shared.tensor([CHANNELS_PER_BLOCK, kernel_height, kernel_width], tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate input positions
    input_start_h = output_row * stride_h - padding_h
    input_start_w = output_col * stride_w - padding_w
    
    # Loop over kernel elements
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input position
            ih = input_start_h + kh * dilation_h
            iw = input_start_w + kw * dilation_w
            
            # Check bounds
            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_idx * (in_channels * input_height * input_width) +
                                  channel_idx * (input_height * input_width) +
                                  ih * input_width + iw)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                                   channel_idx * (kernel_height * kernel_width) +
                                   kh * kernel_width + kw)
                
                acc += input_val * weight_val
    
    # Store result
    if output_idx < output_height * output_width:
        tl.store(output_ptr + 
                batch_idx * (in_channels * output_height * output_width) +
                channel_idx * (output_height * output_width) +
                output_idx, acc)

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of depthwise convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    kernel_height, kernel_width = weight.shape[2], weight.shape[3]
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 4
    OUTPUT_ELEMENTS_PER_BLOCK = 32
    
    # Grid dimensions
    grid_batch = batch_size
    grid_channels = (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK
    grid_output = (output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK
    
    # Launch kernel
    depthwise_conv2d_kernel[(grid_batch, grid_channels, grid_output),](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
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
        in_channels,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        """
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias,
            stride=(self.stride_h, self.stride_w),
            padding=(self.padding_h, self.padding_w),
            dilation=(self.dilation_h, self.dilation_w)
        )

    def extra_repr(self):
        return (
            f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'kernel_size=({self.kernel_size_h}, {self.kernel_size_w}), '
            f'stride=({self.stride_h}, {self.stride_w}), padding=({self.padding_h}, {self.padding_w}), '
            f'dilation=({self.dilation_h}, {self.dilation_w}), groups={self.groups}, bias={self.bias is not None}'
        )