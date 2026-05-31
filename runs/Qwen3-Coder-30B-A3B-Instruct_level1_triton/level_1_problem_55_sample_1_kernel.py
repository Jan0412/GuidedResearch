import torch
import torch.nn as nn
import torch.nn.functional as F
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
    kernel_h,
    kernel_w,
    padding_h,
    padding_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    has_bias,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    
    # Shared memory for input tile
    TILE_H = 16
    TILE_W = 16
    
    # Initialize accumulator
    acc = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
    
    # Loop over kernel elements
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input indices
            ih = tl.arange(0, TILE_H) * stride_h - padding_h + kh * dilation_h
            iw = tl.arange(0, TILE_W) * stride_w - padding_w + kw * dilation_w
            
            # Bounds checking for input dimensions
            ih_valid = (ih >= 0) & (ih < input_height)
            iw_valid = (iw >= 0) & (iw < input_width)
            
            # Load input tile
            input_tile = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
            for c in range(in_channels):
                input_offset = pid_batch * input_stride_0 + c * input_stride_1 + ih * input_stride_2 + iw * input_stride_3
                input_values = tl.load(input_ptr + input_offset, mask=(ih_valid[:, None] & iw_valid[None, :]), other=0.0)
                weight_value = tl.load(weight_ptr + (pid_out_ch * weight_stride_0 + c * weight_stride_1 + kh * weight_stride_2 + kw * weight_stride_3))
                input_tile += input_values * weight_value
            
            acc += input_tile
    
    # Apply bias if needed
    if has_bias:
        bias_value = tl.load(bias_ptr + pid_out_ch)
        acc += bias_value
    
    # Write output
    for oh in range(output_height):
        for ow in range(output_width):
            output_offset = pid_batch * output_stride_0 + pid_out_ch * output_stride_1 + oh * output_stride_2 + ow * output_stride_3
            tl.store(output_ptr + output_offset, acc[oh, ow])

def triton_conv2d(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Triton implementation of 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Compute output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) // stride[1] + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 16
    grid = (
        batch_size,
        out_channels
    )
    
    # Define strides
    input_stride_0 = input_tensor.stride(0)
    input_stride_1 = input_tensor.stride(1)
    input_stride_2 = input_tensor.stride(2)
    input_stride_3 = input_tensor.stride(3)
    
    weight_stride_0 = weight.stride(0)
    weight_stride_1 = weight.stride(1)
    weight_stride_2 = weight.stride(2)
    weight_stride_3 = weight.stride(3)
    
    output_stride_0 = output.stride(0)
    output_stride_1 = output.stride(1)
    output_stride_2 = output.stride(2)
    output_stride_3 = output.stride(3)
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
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
        kernel_h,
        kernel_w,
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
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )

# Note: The above implementation is a simplified version that works for basic cases.
# A full optimized version would require more sophisticated tiling and shared memory usage.