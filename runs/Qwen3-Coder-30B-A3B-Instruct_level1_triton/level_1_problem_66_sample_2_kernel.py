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
    kernel_depth,
    kernel_height,
    kernel_width,
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
    group_size,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_group_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate which output elements this block handles
    output_elements_start = output_idx * OUTPUT_ELEMENTS_PER_BLOCK
    output_elements_end = tl.minimum(output_elements_start + OUTPUT_ELEMENTS_PER_BLOCK, 
                                    output_depth * output_height * output_width)
    
    # Shared memory for input tile
    shared_input = tl.shared_ptr(input_ptr, shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Process multiple output elements per block
    for output_element_offset in range(0, OUTPUT_ELEMENTS_PER_BLOCK, BLOCK_SIZE):
        output_element_idx = output_elements_start + output_element_offset
        
        if output_element_idx >= output_elements_end:
            break
            
        # Calculate output coordinates
        out_z = output_element_idx // (output_height * output_width)
        out_y = (output_element_idx % (output_height * output_width)) // output_width
        out_x = output_element_idx % output_width
        
        # Calculate input coordinates with padding and stride
        in_z_start = out_z * stride_d - padding_d
        in_y_start = out_y * stride_h - padding_h
        in_x_start = out_x * stride_w - padding_w
        
        # Initialize accumulator
        acc = tl.zeros((CHANNELS_PER_BLOCK,), dtype=tl.float32)
        
        # Loop over kernel dimensions
        for k_d in range(kernel_depth):
            for k_h in range(kernel_height):
                for k_w in range(kernel_width):
                    # Calculate input position
                    in_z = in_z_start + k_d * dilation_d
                    in_y = in_y_start + k_h * dilation_h
                    in_x = in_x_start + k_w * dilation_w
                    
                    # Check bounds
                    if (in_z >= 0 and in_z < input_depth and 
                        in_y >= 0 and in_y < input_height and 
                        in_x >= 0 and in_x < input_width):
                        
                        # Calculate weight index
                        weight_idx = channel_group_idx * kernel_depth * kernel_height * kernel_width + \
                                   k_d * kernel_height * kernel_width + \
                                   k_h * kernel_width + \
                                   k_w
                        
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          batch_idx * in_channels * input_depth * input_height * input_width +
                                          channel_group_idx * input_depth * input_height * input_width +
                                          in_z * input_height * input_width +
                                          in_y * input_width +
                                          in_x)
                        
                        # Load weight value
                        weight_val = tl.load(weight_ptr + weight_idx)
                        
                        # Accumulate
                        acc += input_val * weight_val
        
        # Store results
        for c in range(CHANNELS_PER_BLOCK):
            if channel_group_idx * CHANNELS_PER_BLOCK + c < out_channels:
                output_idx_global = batch_idx * out_channels * output_depth * output_height * output_width + \
                                  (channel_group_idx * CHANNELS_PER_BLOCK + c) * output_depth * output_height * output_width + \
                                  out_z * output_height * output_width + \
                                  out_y * output_width + \
                                  out_x
                
                tl.store(output_ptr + output_idx_global, acc[c])

def triton_conv3d(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Triton implementation of 3D convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_height = (input_height + 2 * padding[1] - (dilation[1] * (kernel_height - 1) + 1)) // stride[1] + 1
    output_width = (input_width + 2 * padding[2] - (dilation[2] * (kernel_width - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Handle bias if present
    if bias is not None:
        bias = bias.contiguous()
    
    # Prepare kernel parameters
    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    dilation_d, dilation_h, dilation_w = dilation
    
    # Define block sizes
    BLOCK_SIZE = 16
    CHANNELS_PER_BLOCK = 4
    OUTPUT_ELEMENTS_PER_BLOCK = 32
    
    # Grid configuration
    grid = (
        batch_size,  # batch dimension
        (out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,  # channel groups
        (output_depth * output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK  # output elements
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
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
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
        groups,
        in_channels // groups,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.
    Optimized with custom Triton kernels.
    """
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        """
        # Ensure inputs are contiguous on GPU
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Use our custom Triton implementation
        return triton_conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)