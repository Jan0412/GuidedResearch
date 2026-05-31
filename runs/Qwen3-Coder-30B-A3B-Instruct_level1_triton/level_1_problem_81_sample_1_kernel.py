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
    BLOCK_SIZE_C: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    
    # Calculate output dimensions
    out_h = height_out
    out_w = width_out
    
    # Shared memory for input tile
    tile_size_h = BLOCK_SIZE_H + 2 * padding
    tile_size_w = BLOCK_SIZE_W + 2 * padding
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(0, in_channels, BLOCK_SIZE_C):
        # Load weight slice
        weight_slice = tl.load(weight_ptr + 
                              tl.arange(0, BLOCK_SIZE_C)[None, :] + 
                              tl.arange(0, kernel_size)[:, None] * in_channels * out_channels +
                              tl.arange(0, out_channels)[None, :] * in_channels * kernel_size)
        
        # Load input tile
        input_tile = tl.zeros((tile_size_h, tile_size_w), dtype=tl.float32)
        
        # Load input data with proper indexing
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate source position in input
                src_h = out_h_idx * stride + kh * dilation - padding
                src_w = out_w_idx * stride + kw * dilation - padding
                
                # Check bounds
                if src_h >= 0 and src_h < height_in and src_w >= 0 and src_w < width_in:
                    # Load from input
                    input_val = tl.load(input_ptr + 
                                       batch_idx * in_channels * height_in * width_in +
                                       tl.arange(0, BLOCK_SIZE_C)[None, :] * height_in * width_in +
                                       src_h * width_in + src_w)
                    input_tile[kh:kh+1, kw:kw+1] += input_val
        
        # Perform convolution
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute output position
                out_h_pos = out_h_idx * stride + kh * dilation - padding
                out_w_pos = out_w_idx * stride + kw * dilation - padding
                
                # Check bounds for output
                if out_h_pos >= 0 and out_h_pos < height_out and out_w_pos >= 0 and out_w_pos < width_out:
                    # Accumulate
                    acc[out_h_pos, out_w_pos] += tl.sum(weight_slice * input_tile[kh:kh+1, kw:kw+1])
    
    # Apply bias if enabled
    if bias_enabled:
        bias = tl.load(bias_ptr + tl.arange(0, out_channels))
        acc += bias[None, :]
    
    # Store result
    tl.store(output_ptr + 
             batch_idx * out_channels * height_out * width_out +
             tl.arange(0, out_channels)[:, None] * height_out * width_out +
             out_h_idx * width_out + out_w_idx,
             acc)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Calculate output size
    height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Configure block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    
    # Grid configuration
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
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