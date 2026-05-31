import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
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
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    group_size_in,
    group_size_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program processes one output element
    output_idx = pid
    
    # Convert linear index to 4D coordinates (b, c, h, w)
    output_w = output_idx % output_width
    output_h = (output_idx // output_width) % output_height
    output_c = (output_idx // (output_width * output_height)) % out_channels
    batch_idx = output_idx // (output_width * output_height * out_channels)
    
    # Group assignment
    group_id = output_c // group_size_out
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Handle bias if present
    if bias_ptr is not None:
        acc = tl.load(bias_ptr + output_c)
    
    # Loop over input channels and kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input position
            input_h = output_h * stride_h - padding_h + kh * dilation_h
            input_w = output_w * stride_w - padding_w + kw * dilation_w
            
            # Check bounds
            if input_h >= 0 and input_h < input_height and input_w >= 0 and input_w < input_width:
                # Calculate input index
                input_idx = batch_idx * (in_channels * input_height * input_width) + \
                           (group_id * group_size_in + (output_c % group_size_out)) * (input_height * input_width) + \
                           input_h * input_width + input_w
                
                # Calculate weight index
                weight_idx = group_id * (group_size_out * kernel_height * kernel_width * group_size_in) + \
                            (output_c % group_size_out) * (kernel_height * kernel_width * group_size_in) + \
                            kh * (kernel_width * group_size_in) + \
                            kw * group_size_in + \
                            (output_c % group_size_out) % group_size_in
                
                # Load input and weight values
                input_val = tl.load(input_ptr + input_idx)
                weight_val = tl.load(weight_ptr + weight_idx)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Store result
    output_idx_global = batch_idx * (out_channels * output_height * output_width) + \
                       output_c * (output_height * output_width) + \
                       output_h * output_width + output_w
    tl.store(output_ptr + output_idx_global, acc)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
        """
        batch_size, _, input_height, input_width = x.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        
        # Calculate output dimensions
        output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_height - 1) + 1
        output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + 1
        
        # Ensure tensors are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Calculate total elements in output
        total_elements = batch_size * self.out_channels * output_height * output_width
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Grid configuration
        grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)
        
        # Launch kernel
        if self.bias is not None:
            conv_transpose2d_kernel[grid](
                x, weight, output, self.bias,
                batch_size, self.in_channels, self.out_channels,
                input_height, input_width, output_height, output_width,
                kernel_height, kernel_width,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w,
                self.groups,
                self.in_channels // self.groups,
                self.out_channels // self.groups,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            conv_transpose2d_kernel[grid](
                x, weight, output, None,
                batch_size, self.in_channels, self.out_channels,
                input_height, input_width, output_height, output_width,
                kernel_height, kernel_width,
                stride_h, stride_w, padding_h, padding_w,
                dilation_h, dilation_w,
                self.groups,
                self.in_channels // self.groups,
                self.out_channels // self.groups,
                BLOCK_SIZE=BLOCK_SIZE
            )
        
        return output