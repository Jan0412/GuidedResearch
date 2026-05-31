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
    BLOCK_SIZE: tl.constexpr,
    GROUPS_BLOCK_SIZE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    group_idx = tl.program_id(2)
    
    # Calculate group size
    ch_per_group = out_channels // groups
    
    # Calculate output indices
    out_h = tl.program_id(3)
    out_w = tl.program_id(4)
    
    # Shared memory for weight tile
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(GROUPS_BLOCK_SIZE, k_h, k_w))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel elements
    for i in range(0, in_channels, BLOCK_SIZE):
        # Load weight tile
        weight_offset = group_idx * ch_per_group * in_channels * k_h * k_w + \
                       (out_ch_idx % ch_per_group) * in_channels * k_h * k_w + \
                       i * k_h * k_w
        
        # Load input tile
        input_offset = batch_idx * in_channels * h_in * w_in + \
                      i * h_in * w_in
        
        # Process kernel elements
        for kh in range(k_h):
            for kw in range(k_w):
                # Calculate input position
                ih = out_h * stride_h - padding_h + kh * dilation_h
                iw = out_w * stride_w - padding_w + kw * dilation_w
                
                # Check bounds
                if ih >= 0 and ih < h_in and iw >= 0 and iw < w_in:
                    # Calculate input index
                    input_idx = input_offset + ih * w_in + iw
                    
                    # Calculate weight index
                    weight_idx = weight_offset + kh * k_w + kw
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store result
    if out_h < h_out and out_w < w_out:
        output_idx = batch_idx * out_channels * h_out * w_out + \
                    out_ch_idx * h_out * w_out + \
                    out_h * w_out + out_w
        tl.store(output_ptr + output_idx, acc, mask=True)

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
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get dimensions
        batch_size, _, h_in, w_in = x.shape
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        
        # Calculate output dimensions
        h_out = (h_in - 1) * stride_h - 2 * padding_h + (k_h - 1) * dilation_h + 1 + self.output_padding[0]
        w_out = (w_in - 1) * stride_w - 2 * padding_w + (k_w - 1) * dilation_w + 1 + self.output_padding[1]
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, h_out, w_out, device=x.device, dtype=torch.float32)
        
        # Ensure tensors are contiguous and on correct device
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Launch kernel
        if self.bias is not None:
            bias = self.bias.contiguous()
        else:
            bias = None
            
        # Define grid dimensions
        grid = (
            batch_size,
            self.out_channels,
            self.groups,
            h_out,
            w_out
        )
        
        # Kernel parameters
        BLOCK_SIZE = 32
        GROUPS_BLOCK_SIZE = 32
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x,
            weight,
            output,
            bias,
            batch_size,
            self.in_channels,
            self.out_channels,
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
            self.groups,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUPS_BLOCK_SIZE=GROUPS_BLOCK_SIZE
        )
        
        return output