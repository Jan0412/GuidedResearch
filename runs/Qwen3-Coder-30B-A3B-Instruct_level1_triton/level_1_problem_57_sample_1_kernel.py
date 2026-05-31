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
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    groups,
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
    grid_h = (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_block_alloc((BLOCK_SIZE_H + 2 * padding_h) * (BLOCK_SIZE_W + 2 * padding_w), tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific offsets
        group_in_channels = in_channels // groups
        group_out_channels = out_channels // groups
        
        # Get group-specific pointers
        input_group_ptr = input_ptr + batch_idx * in_channels * input_height * input_width + g * group_in_channels * input_height * input_width
        weight_group_ptr = weight_ptr + g * group_out_channels * group_in_channels * kernel_height * kernel_width
        output_group_ptr = output_ptr + batch_idx * out_channels * output_height * output_width + g * group_out_channels * output_height * output_width
        
        # Process kernel
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input positions
                input_h_start = out_h_idx * stride_h - padding_h + kh
                input_w_start = out_w_idx * stride_w - padding_w + kw
                
                # Check bounds
                if input_h_start >= 0 and input_h_start < input_height and input_w_start >= 0 and input_w_start < input_width:
                    # Load input tile
                    for ih in range(BLOCK_SIZE_H):
                        for iw in range(BLOCK_SIZE_W):
                            h = input_h_start + ih
                            w = input_w_start + iw
                            if h < input_height and w < input_width:
                                input_val = tl.load(input_group_ptr + h * input_width + w)
                                # Accumulate
                                acc[ih, iw] += input_val * tl.load(weight_group_ptr + kh * kernel_width + kw)
    
    # Write output
    for ih in range(BLOCK_SIZE_H):
        for iw in range(BLOCK_SIZE_W):
            h = out_h_idx * BLOCK_SIZE_H + ih
            w = out_w_idx * BLOCK_SIZE_W + iw
            if h < output_height and w < output_width:
                out_val = acc[ih, iw]
                if bias_enabled:
                    out_val += tl.load(bias_ptr + 0)
                tl.store(output_group_ptr + h * output_width + w, out_val)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1):
    """
    Custom Triton implementation of ConvTranspose2d
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding[0]
    output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding[1]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Setup kernel launch parameters
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 8
    
    # Grid configuration
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
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        groups,
        bias is not None,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
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
            
        # Initialize weights using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
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

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_channels={self.in_channels}',
            f'out_channels={self.out_channels}',
            f'kernel_size={self.kernel_size}',
            f'stride={self.stride}',
            f'padding={self.padding}',
            f'output_padding={self.output_padding}',
            f'groups={self.groups}',
            f'bias={self.bias is not None}'
        ])