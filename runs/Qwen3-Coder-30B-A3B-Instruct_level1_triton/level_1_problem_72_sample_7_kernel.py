import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
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
    stride_depth,
    stride_height,
    stride_width,
    padding_depth,
    padding_height,
    padding_width,
    groups,
    bias_enabled,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Calculate group information
    channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    
    # Determine which group this output channel belongs to
    group_idx = out_ch_idx // out_channels_per_group
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input indices
                input_d = out_d_idx * stride_depth - padding_depth + kd
                input_h = out_h_idx * stride_height - padding_height + kh
                input_w = out_w_idx * stride_width - padding_width + kw
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and
                    input_h >= 0 and input_h < input_height and
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input channel index within group
                    in_ch_idx = (out_ch_idx % out_channels_per_group) * channels_per_group
                    
                    # Load input value
                    input_offset = (batch_idx * in_channels * input_depth * input_height * input_width +
                                  in_ch_idx * input_depth * input_height * input_width +
                                  input_d * input_height * input_width +
                                  input_h * input_width +
                                  input_w)
                    input_val = tl.load(input_ptr + input_offset, mask=True)
                    
                    # Load weight value
                    weight_offset = (out_ch_idx * channels_per_group * kernel_depth * kernel_height * kernel_width +
                                   (in_ch_idx % channels_per_group) * kernel_depth * kernel_height * kernel_width +
                                   kd * kernel_height * kernel_width +
                                   kh * kernel_width +
                                   kw)
                    weight_val = tl.load(weight_ptr + weight_offset, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if enabled
    if bias_enabled:
        bias_offset = out_ch_idx
        bias_val = tl.load(bias_ptr + bias_offset, mask=True)
        acc += bias_val
    
    # Write output
    output_offset = (batch_idx * out_channels * output_depth * output_height * output_width +
                    out_ch_idx * output_depth * output_height * output_width +
                    out_d_idx * output_height * output_width +
                    out_h_idx * output_width +
                    out_w_idx)
    tl.store(output_ptr + output_offset, acc, mask=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract dimensions
        batch_size, _, input_depth, input_height, input_width = x.shape
        kernel_depth, kernel_height, kernel_width = self.kernel_size
        stride_depth, stride_height, stride_width = self.stride
        padding_depth, padding_height, padding_width = self.padding
        output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth + kernel_depth + self.output_padding[0]
        output_height = (input_height - 1) * stride_height - 2 * padding_height + kernel_height + self.output_padding[1]
        output_width = (input_width - 1) * stride_width - 2 * padding_width + kernel_width + self.output_padding[2]
        
        # Ensure we're working with contiguous tensors
        x = x.contiguous()
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Prepare pointers
        input_ptr = x.data_ptr()
        weight_ptr = self.weight.data_ptr()
        output_ptr = output.data_ptr()
        bias_ptr = self.bias.data_ptr() if self.bias is not None else 0
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Calculate grid dimensions
        grid = (
            batch_size,
            self.out_channels,
            output_depth,
            output_height,
            output_width
        )
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            input_ptr,
            weight_ptr,
            output_ptr,
            bias_ptr,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            kernel_depth,
            kernel_height,
            kernel_width,
            stride_depth,
            stride_height,
            stride_width,
            padding_depth,
            padding_height,
            padding_width,
            self.groups,
            self.bias is not None,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output