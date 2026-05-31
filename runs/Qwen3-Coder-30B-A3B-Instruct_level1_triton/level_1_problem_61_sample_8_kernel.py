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
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Calculate output position
    out_d = out_d_idx * stride_d - padding_d
    out_h = out_h_idx * stride_h - padding_h
    out_w = out_w_idx * stride_w - padding_w
    
    # Group handling
    group_size = out_channels // groups
    group_idx = out_c_idx // group_size
    local_c_idx = out_c_idx % group_size
    
    # Shared memory for input tile
    shared_input = tl.shared_ptr(input_ptr, shape=[1, 1, 1, 1], dtype=tl.float32)
    shared_weight = tl.shared_ptr(weight_ptr, shape=[1, 1, 1, 1], dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for k_d in range(kernel_depth):
        for k_h in range(kernel_height):
            for k_w in range(kernel_width):
                # Calculate input indices
                in_d = out_d + k_d
                in_h = out_h + k_h
                in_w = out_w + k_w
                
                # Check bounds
                if (in_d >= 0 and in_d < input_depth and 
                    in_h >= 0 and in_h < input_height and 
                    in_w >= 0 and in_w < input_width):
                    
                    # Calculate input index
                    input_idx = (
                        batch_idx * (in_channels * input_depth * input_height * input_width) +
                        (local_c_idx * input_depth * input_height * input_width) +
                        (in_d * input_height * input_width) +
                        (in_h * input_width) +
                        in_w
                    )
                    
                    # Calculate weight index
                    weight_idx = (
                        out_c_idx * (in_channels // groups * kernel_depth * kernel_height * kernel_width) +
                        (local_c_idx * kernel_depth * kernel_height * kernel_width) +
                        (k_d * kernel_height * kernel_width) +
                        (k_h * kernel_width) +
                        k_w
                    )
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store result
    output_idx = (
        batch_idx * (out_channels * output_depth * output_height * output_width) +
        (out_c_idx * output_depth * output_height * output_width) +
        (out_d_idx * output_height * output_width) +
        (out_h_idx * output_width) +
        out_w_idx
    )
    
    tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1):
    """
    Triton implementation of 3D transposed convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_d = (input_depth - 1) * stride_d - 2 * pad_d + kernel_depth + output_padding[0]
    out_h = (input_height - 1) * stride_h - 2 * pad_h + kernel_height + output_padding[1]
    out_w = (input_width - 1) * stride_w - 2 * pad_w + kernel_width + output_padding[2]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define grid dimensions
    grid = (
        batch_size,
        out_channels,
        out_d,
        out_h,
        out_w
    )
    
    # Launch kernel
    BLOCK_SIZE = 128
    GROUPS = groups
    
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        out_d,
        out_h,
        out_w,
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
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS=GROUPS
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )