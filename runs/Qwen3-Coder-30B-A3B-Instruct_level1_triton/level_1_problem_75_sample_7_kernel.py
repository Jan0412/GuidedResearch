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
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    dilation_height,
    dilation_width,
    groups,
    channels_per_group,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)
    
    # Calculate output dimensions
    out_h_start = out_h * stride_height - padding_height
    out_w_start = out_w * stride_width - padding_width
    
    # Shared memory for accumulating results
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over kernel elements
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input positions
            ih = out_h_start + kh * dilation_height
            iw = out_w_start + kw * dilation_width
            
            # Check bounds
            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                # Calculate input index
                input_idx = (
                    batch_idx * (in_channels * input_height * input_width) +
                    group_idx * channels_per_group * input_height * input_width +
                    ih * input_width +
                    iw
                )
                
                # Calculate weight index
                weight_idx = (
                    group_idx * (channels_per_group * out_channels * kernel_height * kernel_width) +
                    kh * (out_channels * kernel_width) +
                    kw * out_channels
                )
                
                # Load input value
                input_val = tl.load(input_ptr + input_idx, mask=True, other=0.0)
                
                # Load weight values for all output channels in this group
                for oc in range(channels_per_group):
                    weight_val = tl.load(weight_ptr + weight_idx + oc * kernel_width, mask=True, other=0.0)
                    acc += input_val * weight_val
                    
    # Store accumulated result
    output_idx = (
        batch_idx * (out_channels * output_height * output_width) +
        group_idx * channels_per_group * output_height * output_width +
        out_h * output_width +
        out_w
    )
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + group_idx * channels_per_group, mask=True, other=0.0)
        acc += bias_val
    
    # Store final result
    tl.store(output_ptr + output_idx, acc, mask=True)


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
        # Get dimensions
        batch_size, _, input_height, input_width = x.shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        padding_height, padding_width = self.padding
        dilation_height, dilation_width = self.dilation
        groups = self.groups
        
        # Calculate output dimensions
        output_height = (input_height - 1) * stride_height - 2 * padding_height + dilation_height * (kernel_height - 1) + 1
        output_width = (input_width - 1) * stride_width - 2 * padding_width + dilation_width * (kernel_width - 1) + 1
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Prepare input and weight tensors
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Define grid
        grid = (
            batch_size,
            groups,
            output_height,
            output_width
        )
        
        # Launch kernel
        BLOCK_SIZE = 32
        conv_transpose2d_kernel[grid](
            x,
            weight,
            output,
            self.bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            dilation_height,
            dilation_width,
            groups,
            self.out_channels // groups,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output