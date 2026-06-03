import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def conv2d_kernel(
    # Pointers to inputs and outputs
    input_ptr, weight_ptr, output_ptr,
    # Dimensions
    N, C, H, W,  # Input dimensions
    F, K_h, K_w,  # Filter dimensions (F = output channels)
    stride_h, stride_w, pad_h, pad_w,
    # Block sizes
    BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_F: tl.constexpr, BLOCK_KH: tl.constexpr, BLOCK_KW: tl.constexpr,
):
    # Get output dimensions
    out_h = (H + 2 * pad_h - K_h) // stride_h + 1
    out_w = (W + 2 * pad_w - K_w) // stride_w + 1
    
    # Program IDs for output
    batch_id = tl.program_id(0)
    out_f_id = tl.program_id(1)  # output channel
    out_h_id = tl.program_id(2)  # output height
    out_w_id = tl.program_id(3)  # output width
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_F], dtype=tl.float32)
    
    # Loop over input channels
    for c in range(C):
        # Loop over kernel height
        for kh in range(K_h):
            # Compute input h position
            in_h = out_h_id * stride_h + kh - pad_h
            if in_h < 0 or in_h >= H:
                continue
                
            # Loop over kernel width
            for kw in range(K_w):
                # Compute input w position
                in_w = out_w_id * stride_w + kw - pad_w
                if in_w < 0 or in_w >= W:
                    continue
                    
                # Compute input pointer offset
                input_offset = (batch_id * C * H * W + 
                               c * H * W + 
                               in_h * W + 
                               in_w)
                
                # Compute weight pointer offset
                weight_offset = (out_f_id * C * K_h * K_w + 
                                c * K_h * K_w + 
                                kh * K_w + 
                                kw)
                
                # Load input and weight values
                x = tl.load(input_ptr + input_offset)
                w = tl.load(weight_ptr + weight_offset)
                
                # Accumulate
                acc += x * w
    
    # Store result
    output_offset = (batch_id * F * out_h * out_w + 
                    out_f_id * out_h * out_w + 
                    out_h_id * out_w + 
                    out_w_id)
    tl.store(output_ptr + output_offset, acc[0])


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, 
                  bias: torch.Tensor = None,
                  stride: Tuple[int, int] = (1, 1), 
                  padding: Tuple[int, int] = (0, 0),
                  dilation: Tuple[int, int] = (1, 1),
                  groups: int = 1) -> torch.Tensor:
    """
    Triton-based Conv2d implementation.
    Note: This is a simplified implementation that works for groups=1.
    """
    assert groups == 1, "Only groups=1 is supported in this implementation"
    assert dilation == (1, 1), "Only dilation=(1,1) is supported in this implementation"
    
    # Ensure inputs are contiguous
    input = input.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    N, C, H, W = input.shape
    F, _, K_h, K_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    # Compute output dimensions
    out_h = (H + 2 * pad_h - K_h) // stride_h + 1
    out_w = (W + 2 * pad_w - K_w) // stride_w + 1
    
    # Prepare output tensor
    output = torch.empty(N, F, out_h, out_w, dtype=input.dtype, device=input.device)
    
    # Define kernel launch parameters
    # We use a grid with dimensions: [batch, output_channels, output_h, output_w]
    grid = (N, F, out_h, out_w)
    
    # Launch kernel
    conv2d_kernel[grid](
        input, weight, output,
        N, C, H, W,
        F, K_h, K_w,
        stride_h, stride_w, pad_h, pad_w,
        BLOCK_C=16, BLOCK_H=8, BLOCK_W=8,
        BLOCK_F=1, BLOCK_KH=4, BLOCK_KW=4
    )
    
    # Add bias if provided
    if bias is not None:
        # Reshape bias for broadcasting: [1, F, 1, 1]
        bias_view = bias.view(1, F, 1, 1)
        output = output + bias_view
    
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, x):
        # Get the conv1 parameters
        weight = self.conv1.weight
        bias = self.conv1.bias
        stride = self.conv1.stride
        padding = self.conv1.padding
        dilation = self.conv1.dilation
        groups = self.conv1.groups
        
        # Use our custom Triton convolution
        x = triton_conv2d(x, weight, bias, stride, padding, dilation, groups)
        return x