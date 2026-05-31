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
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_h_idx = tl.program_id(2)
    
    # Calculate output width index
    output_w_idx = tl.program_id(3)
    
    # Shared memory for input tile
    input_tile = tl.shared_tile(input_ptr, [BLOCK_SIZE, BLOCK_SIZE], [0, 1])
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input positions
            ih = output_h_idx * stride_h + kh * dilation_h - padding_h
            iw = output_w_idx * stride_w + kw * dilation_w - padding_w
            
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
                
                # Accumulate
                acc += input_val * weight_val
    
    # Store result
    if output_h_idx < output_height and output_w_idx < output_width:
        tl.store(output_ptr + 
                batch_idx * (in_channels * output_height * output_width) +
                channel_idx * (output_height * output_width) +
                output_h_idx * output_width + output_w_idx,
                acc)

def triton_depthwise_conv2d(input_tensor, weight, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    kernel_height, kernel_width = weight.shape[2], weight.shape[3]
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 16
    CHANNELS_PER_BLOCK = 1
    
    # Grid configuration
    grid = (
        batch_size,
        in_channels,
        (output_height + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (output_width + BLOCK_SIZE - 1) // BLOCK_SIZE
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
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
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.
    Optimized with custom Triton kernels.
    """
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
        
        # Initialize weight tensor
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use Triton kernel for depthwise convolution
        output = triton_depthwise_conv2d(
            x, 
            self.weight, 
            stride=(self.stride_h, self.stride_w),
            padding=(self.padding_h, self.padding_w),
            dilation=(self.dilation_h, self.dilation_w)
        )
        
        # Add bias if present
        if self.bias is not None:
            # Bias is added per channel, so we need to reshape it properly
            bias_reshaped = self.bias.view(1, -1, 1, 1)
            output = output + bias_reshaped
            
        return output