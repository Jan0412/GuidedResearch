import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out // G, kD, kH, kW)
    bias_ptr,  # Bias tensor: (C_out,) or None
    output_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, C_out, G,  # Batch size, input channels, output channels, groups
    D, H, W,  # Input dimensions
    kD, kH, kW,  # Kernel dimensions
    D_out, H_out, W_out,  # Output dimensions
    s_d, s_h, s_w,  # Strides
    p_d, p_h, p_w,  # Padding
    op_d, op_h, op_w,  # Output padding
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Program IDs: [batch, out_channel_block, depth, height, width]
    batch_idx = tl.program_id(0)
    out_channel_block = tl.program_id(1)
    out_d = tl.program_id(2)
    out_h = tl.program_id(3)
    out_w = tl.program_id(4)
    
    # Calculate actual output channel indices for this block
    out_channel_start = out_channel_block * BLOCK_SIZE_C_OUT
    out_channel_offsets = out_channel_start + tl.arange(0, BLOCK_SIZE_C_OUT)
    out_channel_mask = out_channel_offsets < C_out
    
    # Calculate input position for this output position
    # For transposed convolution: input_pos = (output_pos - output_padding - 1 + padding) // stride + 1
    in_d = (out_d - op_d) // s_d
    in_h = (out_h - op_h) // s_h
    in_w = (out_w - op_w) // s_w
    
    # Check if input position is valid
    valid_input = (in_d >= 0) & (in_d < D) & (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)
    
    # Calculate kernel offsets
    kd_offsets = tl.arange(0, BLOCK_SIZE_KD)
    kh_offsets = tl.arange(0, BLOCK_SIZE_KH)
    kw_offsets = tl.arange(0, BLOCK_SIZE_KW)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=tl.float32)
    
    # Loop over input channels and kernel dimensions
    for c_in in range(C_in):
        # Get input value
        if valid_input:
            in_ptr = x_ptr + batch_idx * (C_in * D * H * W) + c_in * (D * H * W) + in_d * (H * W) + in_h * W + in_w
            x_val = tl.load(in_ptr, mask=True)
        else:
            x_val = 0.0
        
        # Loop over kernel dimensions
        for kd in range(kD):
            for kh in range(kH):
                for kw in range(kW):
                    # Check if kernel position is valid for this output
                    out_d_calc = in_d * s_d + kd - p_d
                    out_h_calc = in_h * s_h + kh - p_h
                    out_w_calc = in_w * s_w + kw - p_w
                    
                    valid_kernel = (out_d_calc == out_d) & (out_h_calc == out_h) & (out_w_calc == out_w)
                    
                    if valid_kernel:
                        # Get weight value
                        # Weight shape: (C_in, C_out // G, kD, kH, kW)
                        # For grouped convolution, map input channel to group
                        group_idx = c_in // (C_in // G)
                        c_in_group = c_in % (C_in // G)
                        
                        w_ptr_offset = c_in * (C_out // G * kD * kH * kW) + c_in_group * (kD * kH * kW) + kd * (kH * kW) + kh * kW + kw
                        w_val = tl.load(w_ptr + w_ptr_offset, mask=True)
                        
                        # Accumulate
                        acc += x_val * w_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_offsets = out_channel_start + tl.arange(0, BLOCK_SIZE_C_OUT)
        bias_mask = bias_offsets < C_out
        bias_val = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
        acc += bias_val
    
    # Store output
    out_ptr_offset = (batch_idx * C_out + out_channel_start) * (D_out * H_out * W_out) + out_d * (H_out * W_out) + out_h * W_out + out_w
    out_store_mask = out_channel_mask
    tl.store(output_ptr + out_ptr_offset, acc.to(tl.float32), mask=out_store_mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """Wrapper function for the conv_transpose3d kernel"""
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract shapes
    B, C_in, D, H, W = x.shape
    C_in2, C_out_per_group, kD, kH, kW = weight.shape
    C_out = C_in2 * C_out_per_group  # Should equal C_in * C_out_per_group for correct shape
    
    # Calculate output dimensions
    D_out = (D - 1) * stride[0] - 2 * padding[0] + kD + output_padding[0]
    H_out = (H - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1]
    W_out = (W - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2]
    
    # Create output tensor
    output = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Kernel launch configuration
    BLOCK_SIZE_C_OUT = 8
    BLOCK_SIZE_C_IN = 4
    BLOCK_SIZE_KD = 3
    BLOCK_SIZE_KH = 3
    BLOCK_SIZE_KW = 3
    
    grid = (B, (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT, D_out, H_out, W_out)
    
    conv_transpose3d_kernel[grid](
        x, weight, bias, output,
        B, C_in, C_out, groups,
        D, H, W,
        kD, kH, kW,
        D_out, H_out, W_out,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        BLOCK_SIZE_KD=BLOCK_SIZE_KD,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Create weight and bias parameters manually
        kD, kH, kW = kernel_size
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, kD, kH, kW))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call our custom Triton implementation
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.output_padding, self.groups
        )