import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,     # Input tensor pointer
    weight_ptr,    # Weight tensor pointer
    output_ptr,    # Output tensor pointer
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
    padding_d,
    padding_w,
    padding_h,
    dilation_d,
    dilation_w,
    dilation_h,
    groups,
    group_size,
    BLOCK_SIZE: tl.constexpr,
    THREADS_PER_BLOCK: tl.constexpr
):
    # Get thread and block indices
    block_id = tl.program_id(0)
    thread_id = tl.thread_id()
    
    # Calculate global output indices
    batch_idx = block_id // (out_channels * output_depth * output_width * output_height)
    remaining = block_id % (out_channels * output_depth * output_width * output_height)
    out_ch_idx = remaining // (output_depth * output_width * output_height)
    remaining = remaining % (output_depth * output_width * output_height)
    out_d_idx = remaining // (output_width * output_height)
    remaining = remaining % (output_width * output_height)
    out_w_idx = remaining // output_height
    out_h_idx = remaining % output_height
    
    # Check bounds
    if batch_idx >= batch_size or out_ch_idx >= out_channels or out_d_idx >= output_depth or out_w_idx >= output_width or out_h_idx >= output_height:
        return
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(THREADS_PER_BLOCK,))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for k_d in range(kernel_depth):
        for k_w in range(kernel_width):
            for k_h in range(kernel_height):
                # Calculate input position
                in_d = out_d_idx * stride_d - padding_d + k_d * dilation_d
                in_w = out_w_idx * stride_w - padding_w + k_w * dilation_w
                in_h = out_h_idx * stride_h - padding_h + k_h * dilation_h
                
                # Check if input position is valid
                if (in_d >= 0 and in_d < input_depth and 
                    in_w >= 0 and in_w < input_width and 
                    in_h >= 0 and in_h < input_height):
                    
                    # Calculate input channel index (considering groups)
                    ch_offset = out_ch_idx // (out_channels // groups) * group_size
                    
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_idx * (in_channels * input_depth * input_width * input_height) +
                                       ch_offset * (input_depth * input_width * input_height) +
                                       in_d * (input_width * input_height) +
                                       in_w * input_height +
                                       in_h)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + 
                                        out_ch_idx * (in_channels // groups * kernel_depth * kernel_width * kernel_height) +
                                        (out_ch_idx % (out_channels // groups)) * (kernel_depth * kernel_width * kernel_height) +
                                        k_d * (kernel_width * kernel_height) +
                                        k_w * kernel_height +
                                        k_h)
                    
                    acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + 
             batch_idx * (out_channels * output_depth * output_width * output_height) +
             out_ch_idx * (output_depth * output_width * output_height) +
             out_d_idx * (output_width * output_height) +
             out_w_idx * output_height +
             out_h_idx, 
             acc)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1):
    """
    Triton implementation of 3D convolution
    """
    batch_size, in_channels, input_depth, input_width, input_height = input_tensor.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    output_height = (input_height + 2 * padding[2] - (dilation[2] * (kernel_height - 1) + 1)) // stride[2] + 1
    
    # Allocate output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_width, output_height, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Set up launch parameters
    total_elements = batch_size * out_channels * output_depth * output_width * output_height
    BLOCK_SIZE = 1024
    THREADS_PER_BLOCK = 256
    
    # Grid size calculation
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv3d_kernel[grid_size](
        input_tensor,
        weight,
        output,
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
        in_channels // groups,
        BLOCK_SIZE=BLOCK_SIZE,
        THREADS_PER_BLOCK=THREADS_PER_BLOCK
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel.
    Optimized using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

# Helper functions to match the original interface
def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    depth = 64
    width = 64
    height = 64
    x = torch.rand(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [3, 64, 3]  # in_channels, out_channels, kernel_size