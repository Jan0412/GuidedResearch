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
    pad_d,
    pad_h,
    pad_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get block IDs
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    channel_id = tl.program_id(2)
    
    # Calculate output indices
    output_idx = tl.program_id(3) * OUTPUT_ELEMENTS_PER_BLOCK
    
    # Shared memory for weight tiles
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(CHANNELS_PER_BLOCK, KERNEL_DEPTH, KERNEL_HEIGHT, KERNEL_WIDTH))
    
    # Process multiple output elements per thread
    for i in range(OUTPUT_ELEMENTS_PER_BLOCK):
        if output_idx + i >= output_depth * output_height * output_width:
            break
            
        # Convert linear output index to 3D coordinates
        out_z = (output_idx + i) // (output_height * output_width)
        out_y = ((output_idx + i) % (output_height * output_width)) // output_width
        out_x = (output_idx + i) % output_width
        
        # Initialize accumulator
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Compute convolution for this output position
        for k_d in range(kernel_depth):
            for k_h in range(kernel_height):
                for k_w in range(kernel_width):
                    # Calculate input coordinates
                    in_z = out_z * stride_d - pad_d + k_d
                    in_y = out_y * stride_h - pad_h + k_h
                    in_x = out_x * stride_w - pad_w + k_w
                    
                    # Check bounds
                    if (in_z >= 0 and in_z < input_depth and 
                        in_y >= 0 and in_y < input_height and 
                        in_x >= 0 and in_x < input_width):
                        
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          batch_id * in_channels * input_depth * input_height * input_width +
                                          channel_id * input_depth * input_height * input_width +
                                          in_z * input_height * input_width +
                                          in_y * input_width +
                                          in_x)
                        
                        # Load weight value
                        weight_val = tl.load(weight_ptr + 
                                           group_id * (out_channels // groups) * in_channels * kernel_depth * kernel_height * kernel_width +
                                           channel_id * kernel_depth * kernel_height * kernel_width +
                                           k_d * kernel_height * kernel_width +
                                           k_h * kernel_width +
                                           k_w)
                        
                        acc += input_val * weight_val
        
        # Add bias if available
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + channel_id)
            acc += bias_val
        
        # Store result
        tl.store(output_ptr + 
                batch_id * out_channels * output_depth * output_height * output_width +
                channel_id * output_depth * output_height * output_width +
                out_z * output_height * output_width +
                out_y * output_width +
                out_x,
                acc)

class ConvTranspose3DTrition(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1, bias=False):
        super(ConvTranspose3DTrition, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size, _, input_depth, input_height, input_width = x.shape
        kernel_depth, kernel_height, kernel_width = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        pad_d, pad_h, pad_w = self.padding
        
        # Calculate output dimensions
        output_depth = (input_depth - 1) * stride_d - 2 * pad_d + kernel_depth + self.output_padding[0]
        output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_height + self.output_padding[1]
        output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_width + self.output_padding[2]
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Launch kernel
        BLOCK_SIZE = 128
        GROUPS_PER_BLOCK = 1
        CHANNELS_PER_BLOCK = 1
        OUTPUT_ELEMENTS_PER_BLOCK = 32
        
        grid = (
            batch_size,  # Batch dimension
            self.groups,  # Group dimension
            self.out_channels,  # Channel dimension
            (output_depth * output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK  # Output elements
        )
        
        # Call the kernel
        conv_transpose3d_kernel[grid](
            x,
            self.weight,
            output,
            self.bias,
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
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            self.groups,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
            OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
        )
        
        return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = ConvTranspose3DTrition(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose3d(x)