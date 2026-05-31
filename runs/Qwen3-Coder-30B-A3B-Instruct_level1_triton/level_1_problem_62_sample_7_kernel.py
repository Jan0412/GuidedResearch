import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_shape,
    weight_shape,
    output_shape,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    weight_height,
    weight_width,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    out_h_id = tl.program_id(2)
    out_w_id = tl.program_id(3)
    
    # Calculate global output index
    output_idx = batch_id * (out_channels * output_height * output_width) + \
                 out_ch_id * (output_height * output_width) + \
                 out_h_id * output_width + out_w_id
    
    # Shared memory for input tile and weight tile
    tile_size = BLOCK_SIZE * BLOCK_SIZE
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(tile_size,))
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(tile_size,))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels (grouped)
    for ch_group in range(0, in_channels, CHANNELS_PER_BLOCK):
        # Calculate channel offset within group
        ch_offset = ch_group
        
        # Calculate input region boundaries for this kernel position
        h_start = out_h_id * stride_h - padding_h
        w_start = out_w_id * stride_w - padding_w
        
        # Initialize accumulator for this group
        group_acc = tl.zeros((1,), dtype=tl.float32)
        
        # Loop over kernel dimensions
        for kh in range(weight_height):
            for kw in range(weight_width):
                # Calculate input position
                ih = h_start + kh * dilation_h
                iw = w_start + kw * dilation_w
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Load input value
                    input_idx = batch_id * (in_channels * input_height * input_width) + \
                                ch_offset * (input_height * input_width) + \
                                ih * input_width + iw
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    
                    # Load weight value
                    weight_idx = out_ch_id * (in_channels // groups * weight_height * weight_width) + \
                                 ch_offset * (weight_height * weight_width) + \
                                 kh * weight_width + kw
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    group_acc += input_val * weight_val
                else:
                    # Out of bounds - contribution is zero
                    pass
        
        # Add group accumulator to total
        acc += group_acc
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_id, mask=True)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + output_idx, acc, mask=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
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
        # Convert to contiguous tensors for better memory access patterns
        x = x.contiguous()
        weight = self.weight.contiguous()
        if self.bias is not None:
            bias = self.bias.contiguous()
        else:
            bias = None
            
        # Get input dimensions
        batch_size, _, input_height, input_width = x.shape
        weight_height, weight_width = self.kernel_size
        
        # Calculate output dimensions
        output_height = (input_height + 2 * self.padding - (self.dilation * (weight_height - 1) + 1)) // self.stride + 1
        output_width = (input_width + 2 * self.padding - (self.dilation * (weight_width - 1) + 1)) // self.stride + 1
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Define block sizes
        BLOCK_SIZE = 16
        CHANNELS_PER_BLOCK = 4
        OUTPUT_ELEMENTS_PER_BLOCK = 16
        
        # Grid configuration
        grid = (
            batch_size,
            self.out_channels,
            output_height,
            output_width
        )
        
        # Launch kernel
        conv2d_kernel[grid](
            x,
            weight,
            output,
            bias,
            x.shape,
            self.weight.shape,
            output.shape,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            self.groups,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            weight_height,
            weight_width,
            BLOCK_SIZE=BLOCK_SIZE,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
            OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
        )
        
        return output