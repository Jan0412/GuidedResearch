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
    group_idx = tl.program_id(3)
    
    # Calculate output position
    out_h_start = out_h_idx * BLOCK_SIZE_H
    out_w_start = out_w_idx * BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_tile(tl.float32, (BLOCK_SIZE_H + 2*padding_h, BLOCK_SIZE_W + 2*padding_w))
    
    # Loop over input channels in tiles
    for c_start in range(0, in_channels, BLOCK_SIZE_C):
        # Load weights for this channel group
        weight = tl.load(weight_ptr + 
                        (group_idx * (out_channels // groups) * in_channels // groups * kernel_h * kernel_w) +
                        (tl.arange(0, BLOCK_SIZE_C)[None, :] + c_start) * kernel_h * kernel_w +
                        tl.arange(0, kernel_h)[:, None] * kernel_w +
                        tl.arange(0, kernel_w)[None, :])
        
        # Load input tile
        input_tile = tl.load(input_ptr + 
                           (batch_idx * in_channels * input_height * input_width) +
                           (c_start + tl.arange(0, BLOCK_SIZE_C)[:, None, None]) * input_height * input_width +
                           (tl.arange(0, BLOCK_SIZE_H)[:, None] * stride_h - padding_h) * input_width +
                           (tl.arange(0, BLOCK_SIZE_W)[None, :] * stride_w - padding_w))
        
        # Perform convolution
        for oh in range(BLOCK_SIZE_H):
            for ow in range(BLOCK_SIZE_W):
                # Calculate output position
                out_h = out_h_start + oh
                out_w = out_w_start + ow
                
                # Check bounds
                if out_h < output_height and out_w < output_width:
                    # Convolution operation
                    acc = tl.zeros((1,), dtype=tl.float32)
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            ih = out_h * stride_h + kh - padding_h
                            iw = out_w * stride_w + kw - padding_w
                            
                            if 0 <= ih < input_height and 0 <= iw < input_width:
                                input_val = input_tile[(ih - (out_h * stride_h - padding_h)), (iw - (out_w * stride_w - padding_w))]
                                weight_val = weight[(kh * kernel_w + kw)]
                                acc += input_val * weight_val
                    
                    # Store result
                    tl.store(output_ptr + 
                            (batch_idx * out_channels * output_height * output_width) +
                            (group_idx * (out_channels // groups) + tl.arange(0, 1)) * output_height * output_width +
                            out_h * output_width + out_w,
                            acc)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1):
    """
    Triton implementation of ConvTranspose2d
    """
    # Extract dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    
    # Initialize output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    
    # Grid configuration
    grid = (
        batch_size,
        (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
        groups
    )
    
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
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
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
        """
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            output_padding=(self.output_padding, self.output_padding),
            groups=self.groups
        )

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, output_padding={self.output_padding}, groups={self.groups}, bias={self.bias is not None}'