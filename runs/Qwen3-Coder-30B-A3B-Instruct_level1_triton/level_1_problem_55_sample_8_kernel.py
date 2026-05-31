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
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    batch_size, in_channels, out_channels, input_height, input_width,
    kernel_height, kernel_width, output_height, output_width,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program processes one output element
    # We'll use a grid of (batch_size * out_channels * output_height * output_width)
    num_programs = batch_size * out_channels * output_height * output_width
    if pid >= num_programs:
        return
    
    # Decompose the linear index
    batch_idx = pid // (out_channels * output_height * output_width)
    remaining = pid % (out_channels * output_height * output_width)
    out_ch_idx = remaining // (output_height * output_width)
    remaining = remaining % (output_height * output_width)
    out_h_idx = remaining // output_width
    out_w_idx = remaining % output_width
    
    # Calculate input coordinates
    input_h_start = out_h_idx * stride_h - padding_h
    input_w_start = out_w_idx * stride_w - padding_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for c in range(in_channels):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position with dilation
                ih = input_h_start + kh * dilation_h
                iw = input_w_start + kw * dilation_w
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                      batch_idx * input_stride_0 +
                                      c * input_stride_1 +
                                      ih * input_stride_2 +
                                      iw * input_stride_3)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr +
                                       out_ch_idx * weight_stride_0 +
                                       c * weight_stride_1 +
                                       kh * weight_stride_2 +
                                       kw * weight_stride_3)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if present
    if has_bias:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        acc += bias_val
    
    # Store output
    tl.store(output_ptr +
             batch_idx * output_stride_0 +
             out_ch_idx * output_stride_1 +
             out_h_idx * output_stride_2 +
             out_w_idx * output_stride_3,
             acc)

def triton_conv2d(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Custom Triton implementation of 2D convolution
    """
    # Ensure inputs are contiguous and on GPU
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Set up grid
    num_programs = batch_size * out_channels * output_height * output_width
    BLOCK_SIZE = 1024
    GRID_SIZE = (num_programs + BLOCK_SIZE - 1) // BLOCK_SIZE
    
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
    if bias is not None:
        has_bias = True
    else:
        has_bias = False
        
    conv2d_kernel[GRID_SIZE](
        input_tensor,
        weight,
        output,
        bias,
        input_stride_0, input_stride_1, input_stride_2, input_stride_3,
        weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
        output_stride_0, output_stride_1, output_stride_2, output_stride_3,
        batch_size, in_channels, out_channels, input_height, input_width,
        kernel_height, kernel_width, output_height, output_width,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        has_bias,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.
    Optimized with custom Triton kernels for better performance.
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
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use custom Triton implementation
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)