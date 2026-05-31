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
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_width,
    input_height,
    output_depth,
    output_width,
    output_height,
    kernel_depth,
    kernel_width,
    kernel_height,
    stride_d,
    stride_w,
    stride_h,
    pad_d,
    pad_w,
    pad_h,
    groups,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    channel_block = tl.program_id(2)
    
    # Calculate output dimensions per group
    channels_per_group = out_channels // groups
    out_channels_per_group = channels_per_group
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Loop over output elements
    for output_idx in range(tl.cdiv(output_depth * output_width * output_height, OUTPUT_ELEMENTS_PER_BLOCK)):
        # Calculate output position
        output_offset = output_idx * OUTPUT_ELEMENTS_PER_BLOCK
        if output_offset >= output_depth * output_width * output_height:
            break
            
        # Calculate output coordinates for this block
        output_d = (output_offset // (output_width * output_height)) % output_depth
        output_w = (output_offset // output_height) % output_width
        output_h = output_offset % output_height
        
        # Initialize accumulator
        acc = tl.zeros((CHANNELS_PER_BLOCK,), dtype=tl.float32)
        
        # For each input channel in this group
        for k in range(in_channels // groups):
            # Calculate input coordinates
            input_d = output_d * stride_d - pad_d
            input_w = output_w * stride_w - pad_w
            input_h = output_h * stride_h - pad_h
            
            # Convolution computation
            for kd in range(kernel_depth):
                for kw in range(kernel_width):
                    for kh in range(kernel_height):
                        # Check bounds
                        input_d_pos = input_d + kd
                        input_w_pos = input_w + kw
                        input_h_pos = input_h + kh
                        
                        # Check if within input bounds
                        if (input_d_pos >= 0 and input_d_pos < input_depth and
                            input_w_pos >= 0 and input_w_pos < input_width and
                            input_h_pos >= 0 and input_h_pos < input_height):
                            
                            # Calculate indices
                            input_idx = (batch_idx * in_channels * input_depth * input_width * input_height +
                                       (group_idx * (in_channels // groups) + k) * input_depth * input_width * input_height +
                                       input_d_pos * input_width * input_height +
                                       input_w_pos * input_height +
                                       input_h_pos)
                            
                            # Load input value
                            input_val = tl.load(input_ptr + input_idx, mask=True)
                            
                            # Calculate weight index
                            weight_idx = (group_idx * out_channels_per_group + channel_block * CHANNELS_PER_BLOCK) * in_channels * kernel_depth * kernel_width * kernel_height + \
                                       k * kernel_depth * kernel_width * kernel_height + \
                                       kd * kernel_width * kernel_height + \
                                       kw * kernel_height + \
                                       kh
                            
                            # Load weight
                            weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                            
                            # Accumulate
                            acc += input_val * weight_val
        
        # Apply bias if available
        if bias_ptr is not None:
            for c in range(CHANNELS_PER_BLOCK):
                if channel_block * CHANNELS_PER_BLOCK + c < out_channels_per_group:
                    bias_idx = group_idx * out_channels_per_group + channel_block * CHANNELS_PER_BLOCK + c
                    acc[c] += tl.load(bias_ptr + bias_idx, mask=True)
        
        # Store result
        for c in range(CHANNELS_PER_BLOCK):
            if channel_block * CHANNELS_PER_BLOCK + c < out_channels_per_group:
                output_idx = (batch_idx * out_channels * output_depth * output_width * output_height +
                            (group_idx * out_channels_per_group + channel_block * CHANNELS_PER_BLOCK + c) * output_depth * output_width * output_height +
                            output_d * output_width * output_height +
                            output_w * output_height +
                            output_h)
                tl.store(output_ptr + output_idx, acc[c], mask=True)

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
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Set up the kernel parameters
        self.kernel_depth, self.kernel_width, self.kernel_height = kernel_size
        self.stride_d, self.stride_w, self.stride_h = stride
        self.pad_d, self.pad_w, self.pad_h = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, input_depth, input_width, input_height = x.shape
        
        # Calculate output dimensions
        output_depth = (input_depth - 1) * self.stride_d - 2 * self.pad_d + self.kernel_depth + self.output_padding[0]
        output_width = (input_width - 1) * self.stride_w - 2 * self.pad_w + self.kernel_width + self.output_padding[1]
        output_height = (input_width - 1) * self.stride_h - 2 * self.pad_h + self.kernel_height + self.output_padding[2]
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_width, output_height, device=x.device, dtype=torch.float32)
        
        # Ensure all tensors are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        if self.bias is not None:
            bias = self.bias.contiguous()
        else:
            bias = None
            
        # Configure kernel launch parameters
        BLOCK_SIZE = 256
        CHANNELS_PER_BLOCK = 8
        OUTPUT_ELEMENTS_PER_BLOCK = 64
        
        # Grid configuration
        grid = (
            batch_size,  # batch dimension
            self.groups,  # group dimension  
            (self.out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK  # channel blocks
        )
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x,
            weight,
            output,
            bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_depth,
            input_width,
            input_height,
            output_depth,
            output_width,
            output_height,
            self.kernel_depth,
            self.kernel_width,
            self.kernel_height,
            self.stride_d,
            self.stride_w,
            self.stride_h,
            self.pad_d,
            self.pad_w,
            self.pad_h,
            self.groups,
            BLOCK_SIZE=BLOCK_SIZE,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
            OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
        )
        
        return output