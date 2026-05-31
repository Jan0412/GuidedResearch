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
    
    # Calculate output dimensions
    grid_h = (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_H + 2 * padding_h, BLOCK_SIZE_W + 2 * padding_w))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate channel offsets
        in_c_offset = g * (in_channels // groups)
        out_c_offset = g * (out_channels // groups)
        
        # Process kernel
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input coordinates
                ih_start = out_h_idx * stride_h + kh - padding_h
                iw_start = out_w_idx * stride_w + kw - padding_w
                
                # Load input tile
                for i in range(BLOCK_SIZE_H):
                    for j in range(BLOCK_SIZE_W):
                        if ih_start + i >= 0 and ih_start + i < input_height and \
                           iw_start + j >= 0 and iw_start + j < input_width:
                            shared_input[i + padding_h, j + padding_w] = tl.load(
                                input_ptr + 
                                batch_idx * input_height * input_width * in_channels +
                                (ih_start + i) * input_width * in_channels +
                                (iw_start + j) * in_channels +
                                in_c_offset
                            )
                        else:
                            shared_input[i + padding_h, j + padding_w] = 0.0
                
                # Compute convolution for this kernel position
                for oc in range(out_channels // groups):
                    # Load weight
                    weight_val = tl.load(weight_ptr + 
                                       kh * kernel_w * out_channels * in_channels +
                                       kw * out_channels * in_channels +
                                       out_c_offset * in_channels +
                                       in_c_offset)
                    
                    # Accumulate
                    for i in range(BLOCK_SIZE_H):
                        for j in range(BLOCK_SIZE_W):
                            acc[i, j] += shared_input[i + padding_h, j + padding_w] * weight_val
    
    # Store output
    for i in range(BLOCK_SIZE_H):
        for j in range(BLOCK_SIZE_W):
            if out_h_idx * BLOCK_SIZE_H + i < output_height and out_w_idx * BLOCK_SIZE_W + j < output_width:
                tl.store(
                    output_ptr + 
                    batch_idx * output_height * output_width * out_channels +
                    (out_h_idx * BLOCK_SIZE_H + i) * output_width * out_channels +
                    (out_w_idx * BLOCK_SIZE_W + j) * out_channels +
                    out_c_offset,
                    acc[i, j]
                )

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1):
    """
    Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 8
    
    grid = (
        batch_size,
        (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
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
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
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
            self.bias = nn.Parameter(torch.zeros(out_channels))
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
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            output_padding=(self.output_padding, self.output_padding),
            groups=self.groups
        )