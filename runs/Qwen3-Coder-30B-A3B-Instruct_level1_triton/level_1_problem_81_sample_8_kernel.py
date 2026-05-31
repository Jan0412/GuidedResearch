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
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_size,
    stride,
    padding,
    dilation,
    bias_enabled,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    
    # Calculate output dimensions
    grid_h = (height_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (width_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_ptr(input_ptr, (BLOCK_SIZE_H, BLOCK_SIZE_W))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(0, in_channels, BLOCK_SIZE_C):
        # Load weight tile
        weight_tile = tl.load(weight_ptr + 
                             tl.arange(0, BLOCK_SIZE_C)[None, :] * out_channels +
                             tl.arange(0, out_channels)[:, None])
        
        # Load input tile
        input_tile = tl.load(input_ptr + 
                            batch_idx * in_channels * height_in * width_in +
                            c * height_in * width_in +
                            tl.arange(0, BLOCK_SIZE_H)[:, None] * width_in +
                            tl.arange(0, BLOCK_SIZE_W)[None, :])
        
        # Compute convolution
        acc += tl.dot(input_tile, weight_tile)
    
    # Apply bias if enabled
    if bias_enabled:
        bias_tile = tl.load(bias_ptr + tl.arange(0, out_channels))
        acc += bias_tile[None, :]
    
    # Write output
    tl.store(output_ptr + 
             batch_idx * out_channels * height_out * width_out +
             tl.arange(0, BLOCK_SIZE_H)[:, None] * width_out +
             tl.arange(0, BLOCK_SIZE_W)[None, :], 
             acc)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Custom Triton implementation of ConvTranspose2d for better performance.
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    
    # Grid dimensions
    grid = (
        batch_size,
        (height_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (width_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_size,
        stride,
        padding,
        dilation,
        bias is not None,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and square kernel, supporting dilation, padding, and stride.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
        
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
            dilation=self.dilation
        )

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.bias is not None}'