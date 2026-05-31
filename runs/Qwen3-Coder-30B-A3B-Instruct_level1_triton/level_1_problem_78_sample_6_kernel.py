import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
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
    pad_h,
    pad_w,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which output channel this program handles
    output_channel = pid
    
    if output_channel >= out_channels:
        return
        
    # Calculate output dimensions
    batch_offset = tl.program_id(1) * in_channels * input_height * input_width
    output_batch_offset = tl.program_id(1) * out_channels * output_height * output_width
    
    # Load bias if available
    bias_val = 0.0
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + output_channel)
    
    # Loop over output spatial dimensions
    for oh in range(output_height):
        for ow in range(output_width):
            # Initialize accumulator
            acc = 0.0
            
            # Loop over input channels and kernel positions
            for ic in range(in_channels):
                for kh in range(kernel_height):
                    for kw in range(kernel_width):
                        # Calculate input position
                        ih = oh * stride_h - pad_h + kh
                        iw = ow * stride_w - pad_w + kw
                        
                        # Check bounds
                        if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                            # Calculate input and weight indices
                            input_idx = batch_offset + ic * input_height * input_width + ih * input_width + iw
                            weight_idx = output_channel * in_channels * kernel_height * kernel_width + \
                                        ic * kernel_height * kernel_width + kh * kernel_width + kw
                            
                            # Load values and accumulate
                            input_val = tl.load(input_ptr + input_idx)
                            weight_val = tl.load(weight_ptr + weight_idx)
                            acc += input_val * weight_val
            
            # Add bias and store result
            acc += bias_val
            output_idx = output_batch_offset + output_channel * output_height * output_width + oh * output_width + ow
            tl.store(output_ptr + output_idx, acc)

def triton_conv_transpose2d(input_tensor, weight, bias, stride=(1,1), padding=(0,0)):
    """
    Triton implementation of 2D transposed convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_height
    output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_width
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous and on GPU
    input_tensor = input_tensor.contiguous().to(torch.float32)
    weight = weight.contiguous().to(torch.float32)
    if bias is not None:
        bias = bias.contiguous().to(torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024
    GROUP_SIZE = 32
    
    # Grid configuration
    grid = (
        out_channels,  # One block per output channel
        batch_size     # One block per batch
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
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
        pad_h,
        pad_w,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding)