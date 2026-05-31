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
    input_row_stride,
    input_col_stride,
    weight_row_stride,
    weight_col_stride,
    output_row_stride,
    output_col_stride,
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
    groups,
    group_size_in,
    group_size_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_BIAS: tl.constexpr
):
    # Get the program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)
    
    # Calculate the starting position for this program
    batch_id = pid_m // (output_height * output_width)
    remaining = pid_m % (output_height * output_width)
    row_id = remaining // output_width
    col_id = remaining % output_width
    
    # Check bounds
    if batch_id >= batch_size or row_id >= output_height or col_id >= output_width:
        return
    
    # Calculate global output position
    output_offset = batch_id * output_row_stride + row_id * output_col_stride + col_id
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific offsets
        group_input_offset = g * group_size_in * input_row_stride
        group_weight_offset = g * group_size_out * weight_row_stride
        group_output_offset = g * group_size_out * output_row_stride
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # Loop over kernel elements
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position
                input_row = row_id * stride_h - padding_h + kh
                input_col = col_id * stride_w - padding_w + kw
                
                # Check if input position is valid
                if input_row >= 0 and input_row < input_height and input_col >= 0 and input_col < input_width:
                    # Calculate input offset
                    input_offset = group_input_offset + input_row * input_row_stride + input_col
                    
                    # Loop over input channels
                    for ic in range(group_size_in):
                        # Calculate weight offset
                        weight_offset = group_weight_offset + ic * weight_col_stride
                        
                        # Load input and weight
                        input_val = tl.load(input_ptr + input_offset + ic * input_row_stride, mask=(input_row < input_height) & (input_col < input_width))
                        weight_val = tl.load(weight_ptr + weight_offset + kh * weight_col_stride + kw * weight_col_stride, mask=(kh < kernel_height) & (kw < kernel_width))
                        
                        # Accumulate
                        acc += input_val * weight_val
        
        # Store results
        for oc in range(group_size_out):
            output_offset = batch_id * output_row_stride + row_id * output_col_stride + col_id + oc * output_row_stride
            if USE_BIAS:
                bias_val = tl.load(bias_ptr + oc)
                acc[0, oc] += bias_val
            tl.store(output_ptr + output_offset, acc[0, oc])

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to contiguous tensors
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Get dimensions
        batch_size, _, input_height, input_width = x.shape
        _, _, kernel_height, kernel_width = weight.shape
        
        # Calculate output dimensions
        output_height = (input_height + 2 * self.padding - (self.dilation * (kernel_height - 1) + 1)) // self.stride + 1
        output_width = (input_width + 2 * self.padding - (self.dilation * (kernel_width - 1) + 1)) // self.stride + 1
        
        # Create output tensor
        output = torch.empty(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Define kernel parameters
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 8
        
        # Calculate grid
        grid = (
            (batch_size * output_height * output_width + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
            (self.out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
            self.groups
        )
        
        # Launch kernel
        conv2d_kernel[grid](
            x,
            weight,
            output,
            self.bias,
            x.stride(2),
            x.stride(3),
            weight.stride(1),
            weight.stride(2),
            output.stride(2),
            output.stride(3),
            batch_size,
            self.in_channels,
            self.out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.groups,
            self.in_channels // self.groups,
            self.out_channels // self.groups,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            self.bias is not None
        )
        
        return output