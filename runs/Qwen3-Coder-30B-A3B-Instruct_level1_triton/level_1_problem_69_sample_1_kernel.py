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
    bias_enabled,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    channel_idx = tl.program_id(2)
    
    # Calculate output dimensions per group
    channels_per_group = out_channels // groups
    output_elements_per_group = output_height * output_width
    
    # Shared memory for weight tiles
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(GROUPS_PER_BLOCK, CHANNELS_PER_BLOCK, kernel_height, kernel_width))
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_ELEMENTS_PER_BLOCK,), dtype=tl.float32)
    
    # Process input channels in chunks
    for ch_block in range(0, in_channels, CHANNELS_PER_BLOCK):
        # Load weight tile
        weight_offset = group_idx * channels_per_group * kernel_height * kernel_width + \
                       ch_block * kernel_height * kernel_width
        
        # Load weights for this group and channel block
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                for c in range(CHANNELS_PER_BLOCK):
                    if ch_block + c < in_channels:
                        shared_weight[group_idx, c, kh, kw] = tl.load(
                            weight_ptr + weight_offset + c * kernel_height * kernel_width + kh * kernel_width + kw
                        )
        
        # Compute convolution for this channel block
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input positions
                input_h_start = tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK) // output_width
                input_w_start = tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK) % output_width
                
                # Apply stride and dilation
                input_h = input_h_start * stride_h - padding_h + kh * dilation_h
                input_w = input_w_start * stride_w - padding_w + kw * dilation_w
                
                # Check bounds
                valid_mask = (input_h >= 0) & (input_h < input_height) & (input_w >= 0) & (input_w < input_width)
                
                # Load input values
                input_offset = batch_idx * in_channels * input_height * input_width + \
                              (ch_block + tl.arange(0, CHANNELS_PER_BLOCK)) * input_height * input_width
                
                for c in range(CHANNELS_PER_BLOCK):
                    if ch_block + c < in_channels:
                        input_values = tl.load(input_ptr + input_offset + c * input_height * input_width + 
                                             input_h * input_width + input_w, mask=valid_mask, other=0.0)
                        weight_value = shared_weight[group_idx, c, kh, kw]
                        acc += input_values * weight_value
    
    # Store output
    output_offset = batch_idx * out_channels * output_height * output_width + \
                   group_idx * channels_per_group * output_height * output_width
    
    if bias_enabled:
        bias_offset = group_idx * channels_per_group
        for c in range(channels_per_group):
            bias_val = tl.load(bias_ptr + bias_offset + c)
            tl.store(output_ptr + output_offset + c * output_height * output_width + tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK), 
                    acc + bias_val, mask=tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK) < output_elements_per_group)
    else:
        tl.store(output_ptr + output_offset + tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK), 
                acc, mask=tl.arange(0, OUTPUT_ELEMENTS_PER_BLOCK) < output_elements_per_group)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
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
        # Extract dimensions
        batch_size, _, input_height, input_width = x.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        
        # Calculate output dimensions
        output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_height - 1) + 1 + self.output_padding[0]
        output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + 1 + self.output_padding[1]
        
        # Ensure we're working with contiguous tensors
        x = x.contiguous()
        weight = self.weight.contiguous()
        if self.bias is not None:
            bias = self.bias.contiguous()
        else:
            bias = None
            
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Configure kernel launch parameters
        BLOCK_SIZE = 128
        GROUPS_PER_BLOCK = min(4, self.groups)
        CHANNELS_PER_BLOCK = min(32, self.in_channels)
        OUTPUT_ELEMENTS_PER_BLOCK = 64
        
        # Grid configuration
        grid = (
            batch_size,
            (self.groups + GROUPS_PER_BLOCK - 1) // GROUPS_PER_BLOCK,
            (self.out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK
        )
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x,
            weight,
            output,
            bias,
            batch_size,
            self.in_channels,
            self.out_channels,
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
            self.groups,
            self.bias is not None,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
            OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
        )
        
        return output