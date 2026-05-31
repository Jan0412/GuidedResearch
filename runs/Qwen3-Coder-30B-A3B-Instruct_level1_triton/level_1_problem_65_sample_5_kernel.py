import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    groups,
    batch_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    out_c_idx = tl.program_id(3)
    
    # Calculate output position
    out_h_start = out_h_idx * BLOCK_SIZE_H
    out_w_start = out_w_idx * BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_H + 2 * padding_h, BLOCK_SIZE_W + 2 * padding_w))
    
    # Load weights for this output channel
    weight_base = weight_ptr + out_c_idx * in_channels * kernel_h * kernel_w
    
    # Loop over input channels and kernel positions
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    for k in range(in_channels):
        # Load input tile with padding
        for i in range(BLOCK_SIZE_H + 2 * padding_h):
            for j in range(BLOCK_SIZE_W + 2 * padding_w):
                h_in = out_h_start + i - padding_h
                w_in = out_w_start + j - padding_w
                
                if 0 <= h_in < input_height and 0 <= w_in < input_width:
                    input_val = tl.load(input_ptr + batch_idx * in_channels * input_height * input_width +
                                       k * input_height * input_width +
                                       h_in * input_width + w_in)
                else:
                    input_val = 0.0
                
                tl.store(shared_input + i * (BLOCK_SIZE_W + 2 * padding_w) + j, input_val)
        
        # Compute convolution for this input channel
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input position for this kernel position
                for i in range(BLOCK_SIZE_H):
                    for j in range(BLOCK_SIZE_W):
                        h_in = out_h_start + i - padding_h + kh * stride_h
                        w_in = out_w_start + j - padding_w + kw * stride_w
                        
                        if 0 <= h_in < input_height and 0 <= w_in < input_width:
                            input_val = tl.load(input_ptr + batch_idx * in_channels * input_height * input_width +
                                               k * input_height * input_width +
                                               h_in * input_width + w_in)
                        else:
                            input_val = 0.0
                        
                        weight_val = tl.load(weight_base + k * kernel_h * kernel_w + kh * kernel_w + kw)
                        acc[i, j] += input_val * weight_val
    
    # Store output
    for i in range(BLOCK_SIZE_H):
        for j in range(BLOCK_SIZE_W):
            if out_h_start + i < output_height and out_w_start + j < output_width:
                out_idx = batch_idx * out_channels * output_height * output_width + \
                         out_c_idx * output_height * output_width + \
                         (out_h_start + i) * output_width + (out_w_start + j)
                
                if bias_ptr is not None:
                    bias_val = tl.load(bias_ptr + out_c_idx)
                    acc[i, j] += bias_val
                
                tl.store(output_ptr + out_idx, acc[i, j])

def triton_conv_transpose2d(input_tensor, weight, bias, stride, padding, output_padding, groups):
    """
    Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Define block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 8
    
    # Grid dimensions
    grid = (
        batch_size,
        (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
        (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        groups,
        batch_size,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.
    Optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.groups
        )

# For compatibility with the original interface
def get_inputs():
    batch_size = 8
    in_channels = 64
    out_channels = 64
    kernel_size = (3, 7)
    width = 512
    height = 512
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [64, 64, (3, 7)]