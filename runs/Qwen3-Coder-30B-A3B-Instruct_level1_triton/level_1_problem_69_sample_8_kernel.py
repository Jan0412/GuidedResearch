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
    h_in,
    w_in,
    h_out,
    w_out,
    k_h,
    k_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate global output position
    out_h_start = out_h_idx * BLOCK_SIZE_H
    out_w_start = out_w_idx * BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_H + 2 * padding_h, BLOCK_SIZE_W + 2 * padding_w))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific indices
        group_in_channels = in_channels // groups
        group_out_channels = out_channels // groups
        group_in_c_start = g * group_in_channels
        group_out_c_start = g * group_out_channels
        
        # Check if this thread should process this output channel
        if out_c_idx >= group_out_c_start and out_c_idx < group_out_c_start + group_out_channels:
            # Load weight for this group and output channel
            weight_offset = group_out_c_start * group_in_channels * k_h * k_w + (out_c_idx - group_out_c_start) * group_in_channels * k_h * k_w
            
            # Process kernel elements
            for kh in range(k_h):
                for kw in range(k_w):
                    # Calculate input coordinates
                    input_h_start = out_h_start * stride_h - padding_h + kh * dilation_h
                    input_w_start = out_w_start * stride_w - padding_w + kw * dilation_w
                    
                    # Load input tile
                    for ih in range(BLOCK_SIZE_H):
                        for iw in range(BLOCK_SIZE_W):
                            h_in_idx = input_h_start + ih
                            w_in_idx = input_w_start + iw
                            
                            # Boundary check
                            if h_in_idx >= 0 and h_in_idx < h_in and w_in_idx >= 0 and w_in_idx < w_in:
                                input_val = tl.load(input_ptr + 
                                                   batch_idx * in_channels * h_in * w_in +
                                                   (group_in_c_start + 0) * h_in * w_in +
                                                   h_in_idx * w_in + w_in_idx)
                                
                                # Load weight
                                weight_val = tl.load(weight_ptr + weight_offset + 
                                                    kh * k_w * group_in_channels + 
                                                    kw * group_in_channels + 0)
                                
                                # Accumulate
                                acc[ih, iw] += input_val * weight_val
    
    # Write output
    for ih in range(BLOCK_SIZE_H):
        for iw in range(BLOCK_SIZE_W):
            if out_h_start + ih < h_out and out_w_start + iw < w_out:
                out_offset = batch_idx * out_channels * h_out * w_out + \
                           out_c_idx * h_out * w_out + \
                           (out_h_start + ih) * w_out + (out_w_start + iw)
                
                # Add bias if present
                bias_val = tl.load(bias_ptr + out_c_idx) if bias_ptr is not None else 0.0
                
                # Store result
                tl.store(output_ptr + out_offset, acc[ih, iw] + bias_val)

def triton_conv_transpose2d(input_tensor, weight, bias, stride, padding, output_padding, dilation, groups):
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, h_in, w_in = input_tensor.shape
    out_channels, _, k_h, k_w = weight.shape
    
    # Calculate output dimensions
    h_out = (h_in - 1) * stride[0] - 2 * padding[0] + (k_h - 1) * dilation[0] + 1 + output_padding[0]
    w_out = (w_in - 1) * stride[1] - 2 * padding[1] + (k_w - 1) * dilation[1] + 1 + output_padding[1]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, h_out, w_out, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        out_channels,
        (h_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (w_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
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
        h_in,
        w_in,
        h_out,
        w_out,
        k_h,
        k_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.dilation, 
            self.groups
        )