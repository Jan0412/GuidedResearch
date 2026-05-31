import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
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
    kernel_d,
    kernel_h,
    kernel_w,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    dilation_d,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    out_channel_id = tl.program_id(2)
    
    # Calculate output dimensions
    output_elements = output_depth * output_height * output_width
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(1, 1, 1, 1))
    
    # Loop over output elements
    for output_idx in range(tl.cdiv(output_elements, OUTPUT_ELEMENTS_PER_BLOCK)):
        # Calculate output position
        output_offset = output_idx * OUTPUT_ELEMENTS_PER_BLOCK
        if output_offset >= output_elements:
            break
            
        # Calculate output indices
        out_z = (output_offset // (output_height * output_width)) % output_depth
        out_y = (output_offset // output_width) % output_height
        out_x = output_offset % output_width
        
        # Skip if out of bounds
        if out_z >= output_depth or out_y >= output_height or out_x >= output_width:
            continue
            
        # Initialize accumulator
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Handle bias
        if bias_ptr is not None:
            acc = tl.load(bias_ptr + out_channel_id)
        
        # Calculate input positions
        input_z_start = out_z * stride_d - padding_d
        input_y_start = out_y * stride_h - padding_h
        input_x_start = out_x * stride_w - padding_w
        
        # Loop over input channels and kernel
        for k_d in range(kernel_d):
            for k_h in range(kernel_h):
                for k_w in range(kernel_w):
                    # Calculate input coordinates
                    input_z = input_z_start + k_d * dilation_d
                    input_y = input_y_start + k_h * dilation_h
                    input_x = input_x_start + k_w * dilation_w
                    
                    # Check bounds
                    if (input_z >= 0 and input_z < input_depth and
                        input_y >= 0 and input_y < input_height and
                        input_x >= 0 and input_x < input_width):
                        
                        # Calculate channel index
                        channel_idx = (out_channel_id * groups + group_id) % in_channels
                        
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          batch_id * (in_channels * input_depth * input_height * input_width) +
                                          channel_idx * (input_depth * input_height * input_width) +
                                          input_z * (input_height * input_width) +
                                          input_y * input_width +
                                          input_x)
                        
                        # Load weight value
                        weight_val = tl.load(weight_ptr + 
                                           out_channel_id * (in_channels * kernel_d * kernel_h * kernel_w) +
                                           channel_idx * (kernel_d * kernel_h * kernel_w) +
                                           k_d * (kernel_h * kernel_w) +
                                           k_h * kernel_w +
                                           k_w)
                        
                        # Accumulate
                        acc += input_val * weight_val
        
        # Store result
        if output_offset < output_elements:
            output_pos = batch_id * (out_channels * output_depth * output_height * output_width) + \
                        out_channel_id * (output_depth * output_height * output_width) + \
                        out_z * (output_height * output_width) + \
                        out_y * output_width + \
                        out_x
            tl.store(output_ptr + output_pos, acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        # Ensure input is on GPU
        if not x.is_cuda:
            x = x.cuda()
            
        # Get dimensions
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        input_depth = x.shape[2]
        input_height = x.shape[3]
        input_width = x.shape[4]
        
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        dilation_d, dilation_h, dilation_w = self.dilation
        
        # Calculate output dimensions
        output_depth = (input_depth + 2 * padding_d - (dilation_d * (kernel_d - 1) + 1)) // stride_d + 1
        output_height = (input_height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
        output_width = (input_width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Launch Triton kernel
        if batch_size > 0 and self.out_channels > 0 and output_depth > 0 and output_height > 0 and output_width > 0:
            # Define grid
            grid = (
                batch_size,
                self.groups,
                self.out_channels
            )
            
            # Set parameters for kernel launch
            BLOCK_SIZE = 128
            CHANNELS_PER_BLOCK = 16
            OUTPUT_ELEMENTS_PER_BLOCK = 32
            
            # Launch kernel
            conv3d_kernel[grid](
                x,
                self.weight,
                output,
                self.bias,
                batch_size,
                in_channels,
                self.out_channels,
                input_depth,
                input_height,
                input_width,
                output_depth,
                output_height,
                output_width,
                kernel_d,
                kernel_h,
                kernel_w,
                stride_d,
                stride_h,
                stride_w,
                padding_d,
                padding_h,
                padding_w,
                dilation_d,
                dilation_h,
                dilation_w,
                self.groups,
                BLOCK_SIZE=BLOCK_SIZE,
                CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
                OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
            )
        
        return output