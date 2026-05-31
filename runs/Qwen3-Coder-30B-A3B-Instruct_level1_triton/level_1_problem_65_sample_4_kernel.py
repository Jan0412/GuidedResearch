import torch
import torch.nn as nn
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
    out_c_idx = tl.program_id(3)
    
    # Calculate output position
    out_h_start = out_h_idx * BLOCK_SIZE_H
    out_w_start = out_w_idx * BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_tile(input_ptr, [BLOCK_SIZE_H, BLOCK_SIZE_W])
    
    # Loop over input channels
    for c in range(0, in_channels, BLOCK_SIZE_C):
        # Load weights for this channel group
        weight = tl.load(weight_ptr + 
                        out_c_idx * in_channels * kernel_h * kernel_w +
                        c * kernel_h * kernel_w)
        
        # Process output positions
        for oh in range(BLOCK_SIZE_H):
            for ow in range(BLOCK_SIZE_W):
                if out_h_start + oh < output_height and out_w_start + ow < output_width:
                    # Calculate input position
                    h_in = (out_h_start + oh - padding_h) // stride_h
                    w_in = (out_w_start + ow - padding_w) // stride_w
                    
                    # Check bounds
                    if (h_in >= 0 and w_in >= 0 and 
                        h_in < input_height and w_in < input_width):
                        
                        # Apply convolution
                        acc = 0.0
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                h_in_pos = h_in * stride_h + kh - padding_h
                                w_in_pos = w_in * stride_w + kw - padding_w
                                
                                if (h_in_pos >= 0 and w_in_pos >= 0 and 
                                    h_in_pos < input_height and w_in_pos < input_width):
                                    input_val = tl.load(input_ptr + 
                                                       batch_idx * in_channels * input_height * input_width +
                                                       c * input_height * input_width +
                                                       h_in_pos * input_width + w_in_pos)
                                    weight_val = tl.load(weight_ptr + 
                                                        out_c_idx * in_channels * kernel_h * kernel_w +
                                                        c * kernel_h * kernel_w +
                                                        kh * kernel_w + kw)
                                    acc += input_val * weight_val
                        
                        # Store result
                        tl.store(output_ptr + 
                                batch_idx * out_channels * output_height * output_width +
                                out_c_idx * output_height * output_width +
                                (out_h_start + oh) * output_width + (out_w_start + ow),
                                acc)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1,1), padding=(0,0), output_padding=(0,0), groups=1):
    """
    Triton implementation of ConvTranspose2d using fused operations.
    """
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 8
    
    # Calculate grid dimensions
    grid_h = (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_c = (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid = (batch_size, grid_h, grid_w, grid_c)
    
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
        stride[0],
        stride[1],
        padding[0],
        padding[1],
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
        
        # Initialize weights and bias
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
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )

# Note: This implementation is simplified for demonstration purposes.
# A full production version would require more sophisticated optimizations
# including proper handling of group convolutions, bias addition, and better memory access patterns.