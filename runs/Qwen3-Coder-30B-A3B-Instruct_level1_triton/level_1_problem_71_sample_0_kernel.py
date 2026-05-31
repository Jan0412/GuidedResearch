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
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    
    # Calculate output position
    out_h_start = out_h_idx * BLOCK_SIZE_H
    out_w_start = out_w_idx * BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_H + 2 * padding_h, BLOCK_SIZE_W + 2 * padding_w))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate channel range for this group
        ch_start = g * (out_channels // groups)
        ch_end = (g + 1) * (out_channels // groups)
        
        # Loop over input channels
        for c in range(in_channels // groups):
            # Load input tile with padding
            for i in range(BLOCK_SIZE_H + 2 * padding_h):
                for j in range(BLOCK_SIZE_W + 2 * padding_w):
                    h = out_h_start + i - padding_h
                    w = out_w_start + j - padding_w
                    
                    if 0 <= h < input_height and 0 <= w < input_width:
                        input_val = tl.load(input_ptr + 
                                          batch_idx * input_height * input_width * (in_channels // groups) +
                                          c * input_height * input_width +
                                          h * input_width + w)
                    else:
                        input_val = 0.0
                    
                    shared_input[i, j] = input_val
            
            # Compute convolution for this channel
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Calculate weight index
                    weight_idx = ch_start * kernel_h * kernel_w * (in_channels // groups) + \
                                c * kernel_h * kernel_w + \
                                kh * kernel_w + kw
                    
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    # Apply convolution
                    for i in range(BLOCK_SIZE_H):
                        for j in range(BLOCK_SIZE_W):
                            h = out_h_start + i
                            w = out_w_start + j
                            
                            # Check if we're within valid output range
                            if 0 <= h < output_height and 0 <= w < output_width:
                                # Calculate corresponding input position
                                input_h = h // stride_h
                                input_w = w // stride_w
                                
                                # Check if this is a valid position
                                if (h % stride_h == 0 and w % stride_w == 0 and 
                                    input_h < input_height and input_w < input_width):
                                    # Add contribution
                                    acc[i, j] += shared_input[i + kh, j + kw] * weight_val
    
    # Store result
    for i in range(BLOCK_SIZE_H):
        for j in range(BLOCK_SIZE_W):
            h = out_h_start + i
            w = out_w_start + j
            
            if 0 <= h < output_height and 0 <= w < output_width:
                output_idx = batch_idx * output_height * output_width * out_channels + \
                           h * output_width * out_channels + \
                           w * out_channels + \
                           ch_start
                
                tl.store(output_ptr + output_idx, acc[i, j])

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose2d
    """
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride - 2 * padding + kernel_h + output_padding
    output_width = (input_width - 1) * stride - 2 * padding + kernel_w + output_padding
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 8
    
    # Grid dimensions
    grid_h = (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid = (batch_size, grid_h, grid_w)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride,
        stride,
        padding,
        padding,
        groups,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.
    Optimized with Triton kernels for better performance.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=self.groups
        )