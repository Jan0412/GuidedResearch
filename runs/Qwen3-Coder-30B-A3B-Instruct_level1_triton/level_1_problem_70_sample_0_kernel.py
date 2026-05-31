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
    dilation_d,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Calculate output position
    out_d = out_d_idx * stride_d
    out_h = out_h_idx * stride_h
    out_w = out_w_idx * stride_w
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Loop over groups
    for g in range(GROUPS):
        # Calculate group offsets
        group_in_channels = in_channels // groups
        group_out_channels = out_channels // groups
        
        # Calculate channel offsets
        in_ch_offset = g * group_in_channels
        out_ch_offset = g * group_out_channels
        
        # Check if this thread should compute this output
        if out_ch_idx >= group_out_channels:
            continue
            
        # Initialize accumulator
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Loop over kernel dimensions
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input coordinates
                    input_d = (out_d - padding_d + kd * dilation_d)
                    input_h = (out_h - padding_h + kh * dilation_h)
                    input_w = (out_w - padding_w + kw * dilation_w)
                    
                    # Check bounds
                    if (input_d >= 0 and input_d < input_depth and
                        input_h >= 0 and input_h < input_height and
                        input_w >= 0 and input_w < input_width):
                        
                        # Calculate input index
                        input_idx = (batch_idx * (in_channels * input_depth * input_height * input_width) +
                                   (in_ch_offset + (kd * kernel_height * kernel_width + kh * kernel_width + kw) % group_in_channels) *
                                   (input_depth * input_height * input_width) +
                                   input_d * (input_height * input_width) +
                                   input_h * input_width +
                                   input_w)
                        
                        # Calculate weight index
                        weight_idx = (out_ch_offset + out_ch_idx) * (group_in_channels * kernel_depth * kernel_height * kernel_width) + \
                                   (in_ch_offset + (kd * kernel_height * kernel_width + kh * kernel_width + kw) % group_in_channels) * \
                                   (kernel_depth * kernel_height * kernel_width) + \
                                   kd * (kernel_height * kernel_width) + \
                                   kh * kernel_width + \
                                   kw
                        
                        # Load input and weight
                        input_val = tl.load(input_ptr + input_idx, mask=True)
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                        
                        # Accumulate
                        acc += input_val * weight_val
        
        # Store result
        if out_ch_idx < group_out_channels:
            output_idx = (batch_idx * (out_channels * output_depth * output_height * output_width) +
                         (out_ch_offset + out_ch_idx) * (output_depth * output_height * output_width) +
                         out_d_idx * (output_height * output_width) +
                         out_h_idx * output_width +
                         out_w_idx)
            tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), 
                           output_padding=(0, 0, 0), dilation=(1, 1, 1), groups=1):
    """
    Triton implementation of ConvTranspose3d
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    dilation_d, dilation_h, dilation_w = dilation
    
    output_depth = (input_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_depth - 1) + 1 + output_padding[0]
    output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_height - 1) + 1 + output_padding[1]
    output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + 1 + output_padding[2]
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block size and grid
    BLOCK_SIZE = 16
    GRID = (
        batch_size,
        out_channels,
        math.ceil(output_depth / stride_d),
        math.ceil(output_height / stride_h),
        math.ceil(output_width / stride_w)
    )
    
    # Launch kernel
    conv_transpose3d_kernel[GRID](
        input_tensor,
        weight,
        output,
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
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS=groups
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )