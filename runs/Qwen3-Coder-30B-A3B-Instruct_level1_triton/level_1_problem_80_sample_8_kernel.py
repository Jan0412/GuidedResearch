import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,     # Input tensor pointer
    weight_ptr,    # Weight tensor pointer  
    output_ptr,    # Output tensor pointer
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_o, weight_stride_i, weight_stride_h, weight_stride_w,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    batch_size, in_channels, out_channels, input_h, input_w,
    kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
    BLOCK_SIZE: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate output dimensions
    output_h = (input_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    output_w = (input_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    
    # Check bounds
    if out_h_idx >= output_h or out_w_idx >= output_w:
        return
        
    # Shared memory for input tile
    input_tile = tl.shared_pointer(input_ptr + batch_idx * input_stride_n + out_h_idx * stride_h * input_stride_h + out_w_idx * stride_w * input_stride_w, 
                                   (TILE_H + 2 * pad_h, TILE_W + 2 * pad_w))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over input channels
    for c in range(0, in_channels, BLOCK_SIZE):
        # Load weights for this channel
        weight_offset = out_channel_idx * weight_stride_o + c * weight_stride_i
        weight = tl.load(weight_ptr + weight_offset + tl.arange(0, BLOCK_SIZE) * weight_stride_i, mask=(c + tl.arange(0, BLOCK_SIZE) < in_channels))
        
        # Compute convolution for this channel
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input position considering dilation and padding
                ih = out_h_idx * stride_h + kh * dilation_h - pad_h
                iw = out_w_idx * stride_w + kw * dilation_w - pad_w
                
                # Check if input position is valid
                if ih >= 0 and ih < input_h and iw >= 0 and iw < input_w:
                    # Load input value
                    input_val = tl.load(input_ptr + batch_idx * input_stride_n + c * input_stride_c + ih * input_stride_h + iw * input_stride_w)
                    # Multiply with weight and accumulate
                    acc += input_val * weight[kh * kernel_w + kw]
                    
    # Store result
    output_offset = batch_idx * output_stride_n + out_channel_idx * output_stride_c + out_h_idx * output_stride_h + out_w_idx * output_stride_w
    tl.store(output_ptr + output_offset, acc)

def triton_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Triton implementation of 2D convolution using shared memory tiling
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_h, input_w = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_h = (input_h + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) // stride[0] + 1
    output_w = (input_w + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_h, output_w, device=input_tensor.device, dtype=torch.float32)
    
    # Define constants
    BLOCK_SIZE = 32
    TILE_H = 32
    TILE_W = 32
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        (output_h + TILE_H - 1) // TILE_H,
        (output_w + TILE_W - 1) // TILE_W
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, in_channels, out_channels, input_h, input_w,
        kernel_h, kernel_w, padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1],
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_H=TILE_H,
        TILE_W=TILE_W
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.
    Optimized with custom Triton kernels for improved performance.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel for speedup.
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.bias is not None}'