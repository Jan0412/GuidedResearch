import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # Input: (B, C_in, D, H, W)
    w_ptr,  # Weight: (C_in, C_out // groups, kD, kH, kW)
    b_ptr,  # Bias: (C_out,) or None
    out_ptr,  # Output: (B, C_out, D_out, H_out, W_out)
    # Dimensions
    B, C_in, C_out, groups,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    # Strides for memory access
    stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_w_ic, stride_w_oc, stride_w_kd, stride_w_kh, stride_w_kw,
    stride_out_b, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    # Block sizes
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    bid = tl.program_id(0)  # batch index
    cid = tl.program_id(1)  # output channel index (within group)
    did = tl.program_id(2)  # output depth index
    hid = tl.program_id(3)  # output height index
    wid = tl.program_id(4)  # output width index
    
    # Calculate actual output channel
    c_out = cid * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out < C_out
    
    # Calculate input position that contributes to this output
    # For transposed convolution: out_pos = in_pos * stride + (k_pos - 1 - pad) + output_pad
    # So: in_pos = (out_pos - (k_pos - 1 - pad) - output_pad) / stride
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for c_in in range(C_in):
        for kd in range(kD):
            for kh in range(kH):
                for kw in range(kW):
                    # Calculate corresponding input position
                    d_in = (did - (kd - pad_d) - output_pad_d) // stride_d
                    h_in = (hid - (kh - pad_h) - output_pad_h) // stride_h
                    w_in_pos = (wid - (kw - pad_w) - output_pad_w) // stride_w
                    
                    # Check if input position is valid
                    valid = (d_in >= 0) & (d_in < D_in) & \
                            (h_in >= 0) & (h_in < H_in) & \
                            (w_in_pos >= 0) & (w_in_pos < W_in) & \
                            ((did - (kd - pad_d) - output_pad_d) % stride_d == 0) & \
                            ((hid - (kh - pad_h) - output_pad_h) % stride_h == 0) & \
                            ((wid - (kw - pad_w) - output_pad_w) % stride_w == 0)
                    
                    if tl.sum(valid) > 0:
                        # Load input value
                        x_offset = bid * stride_x_b + c_in * stride_x_c + d_in * stride_x_d + h_in * stride_x_h + w_in_pos * stride_x_w
                        x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
                        
                        # Load weight value
                        # Weight shape: (C_in, C_out // groups, kD, kH, kW)
                        # But we need to handle groups
                        group_size = C_out // groups
                        group_id = c_out // group_size
                        group_c_out = c_out % group_size
                        
                        # Check if current output channels belong to this input channel group
                        group_mask = (group_id == c_in)  # For groups > 1, input channel only connects to specific output channels
                        
                        w_offset = c_in * stride_w_ic + group_c_out * stride_w_oc + kd * stride_w_kd + kh * stride_w_kh + kw * stride_w_kw
                        w_val = tl.load(w_ptr + w_offset, mask=group_mask & c_out_mask, other=0.0)
                        
                        # Accumulate
                        acc += tl.where(valid & c_out_mask, x_val[:, None] * w_val, 0.0).sum(axis=1)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out, mask=c_out_mask, other=0.0)
        acc += bias
    
    # Store result
    out_offset = bid * stride_out_b + c_out * stride_out_c + did * stride_out_d + hid * stride_out_h + wid * stride_out_w
    tl.store(out_ptr + out_offset, acc.to(tl.float32), mask=c_out_mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """Custom Triton implementation of 3D transposed convolution"""
    B, C_in, D_in, H_in, W_in = x.shape
    C_in2, C_out_per_group, kD, kH, kW = weight.shape
    C_out = C_in2 * C_out_per_group  # Should equal C_in * (C_out // C_in) for standard case
    
    # Calculate output dimensions
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + (kD - 1) + output_padding[0] + 1
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + (kH - 1) + output_padding[1] + 1
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + (kW - 1) + output_padding[2] + 1
    
    # Create output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Prepare inputs (ensure contiguous)
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None
    
    # Get strides
    stride_x = x.stride()
    stride_w = weight.stride()
    stride_out = out.stride()
    
    # Grid dimensions
    BLOCK_SIZE_C_OUT = 16  # Tunable
    BLOCK_SIZE_K = 4  # Tunable
    
    grid = (B, 
            (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT,
            D_out, H_out, W_out)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, groups,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        stride_x[0], stride_x[1], stride_x[2], stride_x[3], stride_x[4],
        stride_w[0], stride_w[1], stride_w[2], stride_w[3], stride_w[4],
        stride_out[0], stride_out[1], stride_out[2], stride_out[3], stride_out[4],
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                                   stride=stride, padding=padding, 
                                                   output_padding=output_padding, 
                                                   groups=groups, bias=bias)
        # Copy parameters from original layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights and bias
        self.weight = self.conv_transpose3d.weight
        if bias:
            self.bias = self.conv_transpose3d.bias
        else:
            self.register_buffer('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use custom Triton kernel instead of PyTorch implementation
        return triton_conv_transpose3d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.output_padding, 
            self.groups
        )