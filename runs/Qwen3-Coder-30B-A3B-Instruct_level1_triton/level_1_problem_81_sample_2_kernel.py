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
    kernel_size,
    stride,
    padding,
    dilation,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    
    # Calculate output width index
    out_w_idx = tl.program_id(3)
    
    # Shared memory for input tile
    TILE_H = 16
    TILE_W = 16
    
    # Initialize accumulator
    acc = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
    
    # Loop over input channels
    for ch in range(0, in_channels, BLOCK_SIZE):
        # Load weight slice
        weight_block = tl.load(weight_ptr + 
                              out_ch_idx * in_channels * kernel_size * kernel_size +
                              ch * kernel_size * kernel_size +
                              tl.arange(0, kernel_size)[:, None] * kernel_size +
                              tl.arange(0, kernel_size)[None, :])
        
        # Load input tile
        input_tile = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Calculate input position
                h_offset = out_h_idx * stride - padding + i * dilation
                w_offset = out_w_idx * stride - padding + j * dilation
                
                # Check bounds
                if h_offset >= 0 and h_offset < input_height and w_offset >= 0 and w_offset < input_width:
                    input_val = tl.load(input_ptr + 
                                       batch_idx * in_channels * input_height * input_width +
                                       ch * input_height * input_width +
                                       h_offset * input_width + w_offset)
                    input_tile += input_val * weight_block[i, j]
        
        acc += input_tile
    
    # Store output
    output_idx = batch_idx * out_channels * output_height * output_width + \
                 out_ch_idx * output_height * output_width + \
                 out_h_idx * output_width + out_w_idx
    tl.store(output_ptr + output_idx, acc)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    
    # Allocate output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define grid
    grid = (
        batch_size,
        out_channels,
        output_height,
        output_width
    )
    
    # Launch kernel
    BLOCK_SIZE = 16
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
        kernel_size,
        stride,
        padding,
        dilation,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
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
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

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