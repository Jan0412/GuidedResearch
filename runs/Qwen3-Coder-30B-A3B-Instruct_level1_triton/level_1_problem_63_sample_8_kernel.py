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
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_stride_3,
    weight_stride_0,
    weight_stride_1,
    weight_stride_2,
    weight_stride_3,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    padding_h,
    padding_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    has_bias,
    BLOCK_SIZE: tl.constexpr,
):
    # Get block indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_y = tl.program_id(2)
    out_x = tl.program_id(3)
    
    # Calculate output position
    if out_y >= output_height or out_x >= output_width:
        return
    
    # Shared memory for input tile and weight tile
    input_tile = tl.shared_ptr(input_ptr + batch_idx * input_stride_0 + out_ch_idx * input_stride_1, 
                              (input_height + 2 * padding_h, input_width + 2 * padding_w), 
                              (input_stride_2, input_stride_3))
    weight_tile = tl.shared_ptr(weight_ptr + out_ch_idx * weight_stride_0 + out_ch_idx * weight_stride_1, 
                               (kernel_height, kernel_width), 
                               (weight_stride_2, weight_stride_3))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Compute convolution
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input positions
            ih = out_y * stride_h + kh * dilation_h - padding_h
            iw = out_x * stride_w + kw * dilation_w - padding_w
            
            # Check bounds
            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                # Load input value
                input_val = tl.load(input_ptr + batch_idx * input_stride_0 + 
                                  out_ch_idx * input_stride_1 + 
                                  ih * input_stride_2 + 
                                  iw * input_stride_3)
                # Load weight value
                weight_val = tl.load(weight_ptr + out_ch_idx * weight_stride_0 + 
                                   out_ch_idx * weight_stride_1 + 
                                   kh * weight_stride_2 + 
                                   kw * weight_stride_3)
                acc += input_val * weight_val
    
    # Add bias if present
    if has_bias:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + batch_idx * output_stride_0 + 
            out_ch_idx * output_stride_1 + 
            out_y * output_stride_2 + 
            out_x * output_stride_3, 
            acc)

def triton_conv2d(input_tensor, weight, bias, stride, padding, dilation, groups):
    # Handle group convolution
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Set up grid
    grid = (
        batch_size,
        out_channels,
        output_height,
        output_width
    )
    
    # Define block size
    BLOCK_SIZE = 16
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_tensor.stride(0),
        input_tensor.stride(1),
        input_tensor.stride(2),
        input_tensor.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        padding[0],
        padding[1],
        stride[0],
        stride[1],
        dilation[0],
        dilation[1],
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights with Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)