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
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_ch_idx = tl.program_id(2)
    
    # Calculate output dimensions
    output_elements = output_depth * output_height * output_width
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, CHANNELS_PER_BLOCK))
    
    # Process output elements in chunks
    for out_elem_start in range(0, output_elements, OUTPUT_ELEMENTS_PER_BLOCK):
        # Each thread processes one output element
        out_elem_idx = out_elem_start + tl.thread_id(0)
        
        if out_elem_idx >= output_elements:
            return
            
        # Convert linear index to 3D coordinates
        out_d = out_elem_idx // (output_height * output_width)
        remaining = out_elem_idx % (output_height * output_width)
        out_h = remaining // output_width
        out_w = remaining % output_width
        
        # Calculate input start positions
        in_d_start = out_d * stride_d - padding_d
        in_h_start = out_h * stride_h - padding_h
        in_w_start = out_w * stride_w - padding_w
        
        # Initialize accumulator
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Group-based processing
        group_start = group_idx * (out_channels // groups)
        group_end = (group_idx + 1) * (out_channels // groups)
        
        # Only process relevant output channels
        if out_ch_idx < group_start or out_ch_idx >= group_end:
            return
            
        # Loop over input channels and kernel elements
        for ch in range(in_channels):
            # Check if this channel belongs to current group
            if ch % groups != group_idx:
                continue
                
            # Loop over kernel dimensions
            for kd in range(kernel_d):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        # Calculate input position
                        in_d = in_d_start + kd * dilation_d
                        in_h = in_h_start + kh * dilation_h
                        in_w = in_w_start + kw * dilation_w
                        
                        # Check bounds
                        if (in_d >= 0 and in_d < input_depth and
                            in_h >= 0 and in_h < input_height and
                            in_w >= 0 and in_w < input_width):
                            
                            # Calculate input index
                            input_idx = (batch_idx * (in_channels * input_depth * input_height * input_width) +
                                       ch * (input_depth * input_height * input_width) +
                                       in_d * (input_height * input_width) +
                                       in_h * input_width +
                                       in_w)
                            
                            # Calculate weight index
                            weight_idx = (out_ch_idx * (in_channels // groups * kernel_d * kernel_h * kernel_w) +
                                        ch * (kernel_d * kernel_h * kernel_w) +
                                        kd * (kernel_h * kernel_w) +
                                        kh * kernel_w +
                                        kw)
                            
                            # Load input value
                            input_val = tl.load(input_ptr + input_idx, mask=True)
                            
                            # Load weight value
                            weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                            
                            # Accumulate
                            acc += input_val * weight_val
        
        # Apply bias if exists
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + out_ch_idx, mask=True)
            acc += bias_val
            
        # Write output
        output_idx = (batch_idx * (out_channels * output_depth * output_height * output_width) +
                     out_ch_idx * (output_depth * output_height * output_width) +
                     out_d * (output_height * output_width) +
                     out_h * output_width +
                     out_w)
        
        tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv3d(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Triton implementation of 3D convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_d - 1) + 1)) // stride[0] + 1
    output_height = (input_height + 2 * padding[1] - (dilation[1] * (kernel_h - 1) + 1)) // stride[1] + 1
    output_width = (input_width + 2 * padding[2] - (dilation[2] * (kernel_w - 1) + 1)) // stride[2] + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Grid configuration
    grid = (
        batch_size,           # batch dimension
        groups,               # group dimension
        out_channels          # output channel dimension
    )
    
    # Kernel launch parameters
    BLOCK_SIZE = 32
    CHANNELS_PER_BLOCK = 16
    OUTPUT_ELEMENTS_PER_BLOCK = 16
    
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
        kernel_d,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.
    Optimized using custom Triton kernels.
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )