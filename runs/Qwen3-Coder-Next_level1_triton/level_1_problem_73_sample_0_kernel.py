import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to tensors
    X_ptr,  # Input: (B, C_in, D, H, W)
    W_ptr,  # Weight: (C_in, C_out // groups, Kd, Kh, Kw)
    B_ptr,  # Bias: (C_out,) - optional
    Y_ptr,  # Output: (B, C_out, D_out, H_out, W_out)
    # Shapes and strides
    B, C_in, C_out, groups,
    D, H, W,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    stride_out_d, stride_out_h, stride_out_w,
    # Meta-parameters
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Compute output position
    pid_c_out = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Calculate output indices
    c_out_start = pid_c_out * BLOCK_SIZE_C_OUT
    d_out = pid_d * BLOCK_SIZE_D
    h_out = pid_h * BLOCK_SIZE_H
    w_out = pid_w * BLOCK_SIZE_W

    # Compute input position in output tensor
    c_out_range = tl.arange(0, BLOCK_SIZE_C_OUT)
    d_out_range = tl.arange(0, BLOCK_SIZE_D)
    h_out_range = tl.arange(0, BLOCK_SIZE_H)
    w_out_range = tl.arange(0, BLOCK_SIZE_W)

    # Broadcast for output shape
    c_out_idx = c_out_start + c_out_range[:, None, None, None]
    d_out_idx = d_out + d_out_range[None, :, None, None]
    h_out_idx = h_out + h_out_range[None, None, :, None]
    w_out_idx = w_out + w_out_range[None, None, None, :]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C_OUT, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Compute convolution: for each output position, accumulate over input channels and kernel
    # We loop over groups first
    for g in range(groups):
        # Compute input channel range for this group
        c_in_start = g * (C_in // groups)
        
        # Loop over input channels in blocks
        for c_in_offset in range(0, C_in // groups, BLOCK_SIZE_C_IN):
            c_in_idx = c_in_start + c_in_offset + tl.arange(0, BLOCK_SIZE_C_IN)[:, None, None, None]
            
            # Compute corresponding input position for each output position
            # Transposed convolution: output_pos = input_pos * stride - pad + out_pad + kernel_pos
            # => input_pos = output_pos - kernel_pos + pad - out_pad (then divided by stride)
            # For each kernel position (kd, kh, kw)
            
            for kd in range(Kd):
                for kh in range(Kh):
                    for kw in range(Kw):
                        # Calculate input position
                        d_in = d_out_idx - kd + pad_d - out_pad_d
                        h_in = h_out_idx - kh + pad_h - out_pad_w
                        w_in = w_out_idx - kw + pad_w - out_pad_w
                        
                        # Only valid if divisible by stride and within bounds
                        d_valid = (d_in % stride_d == 0)
                        h_valid = (h_in % stride_h == 0)
                        w_valid = (w_in % stride_w == 0)
                        
                        d_in = d_in // stride_d
                        h_in = h_in // stride_h
                        w_in = w_in // stride_w
                        
                        # Check bounds
                        mask_valid = (d_in >= 0) & (d_in < D) & (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                        
                        # Create mask for both output and input
                        output_mask = (c_out_idx < C_out) & (d_out_idx < Y.shape[2]) & (h_out_idx < Y.shape[3]) & (w_out_idx < Y.shape[4])
                        input_mask = mask_valid & (c_in_idx < C_in) & (d_in >= 0) & (d_in < D) & (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                        
                        # Load input value
                        x_offset = pid_b * (C_in * D * H * W) + c_in_idx * (D * H * W) + d_in * (H * W) + h_in * W + w_in
                        x = tl.load(X_ptr + x_offset, mask=input_mask & (c_in_idx < C_in), other=0.0)
                        
                        # Load weight value: weight shape is (C_in, C_out // groups, Kd, Kh, Kw)
                        # Group offset for weights
                        c_out_group_idx = c_out_idx - g * (C_out // groups)
                        w_offset = c_in_idx * (C_out * Kd * Kh * Kw) + c_out_group_idx * (Kd * Kh * Kw) + kd * (Kh * Kw) + kh * Kw + kw
                        w = tl.load(W_ptr + w_offset, mask=input_mask & (c_out_group_idx < C_out // groups), other=0.0)
                        
                        # Accumulate
                        acc += x * w * (input_mask & (c_out_group_idx < C_out // groups)).to(tl.float32)

    # Add bias if present
    if B_ptr is not None:
        b = tl.load(B_ptr + c_out_idx, mask=c_out_idx < C_out, other=0.0)
        acc += b

    # Store result
    y_offset = pid_b * (C_out * Y.shape[2] * Y.shape[3] * Y.shape[4]) + c_out_idx * (Y.shape[2] * Y.shape[3] * Y.shape[4]) + d_out_idx * (Y.shape[3] * Y.shape[4]) + h_out_idx * Y.shape[4] + w_out_idx
    tl.store(Y_ptr + y_offset, acc.to(Y_ptr.dtype.element_ty), mask=(c_out_idx < C_out) & (d_out_idx < Y.shape[2]) & (h_out_idx < Y.shape[3]) & (w_out_idx < Y.shape[4]))


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract shapes
    B, C_in, D, H, W = x.shape
    C_in_w, C_out_per_group, Kd, Kh, Kw = weight.shape
    C_out = C_in_w * C_out_per_group  # Should match C_in * (C_out // groups) with groups
    
    # Compute output shape
    D_out = (D - 1) * stride - 2 * padding + output_padding + (Kd - 1) + 1
    H_out = (H - 1) * stride - 2 * padding + output_padding + (Kh - 1) + 1
    W_out = (W - 1) * stride - 2 * padding + output_padding + (Kw - 1) + 1
    
    # Create output tensor
    y = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    BLOCK_SIZE_C_OUT = 8
    BLOCK_SIZE_C_IN = 8
    BLOCK_SIZE_D = 4
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 16
    
    grid = (
        triton.cdiv(C_out, BLOCK_SIZE_C_OUT),
        B,
        triton.cdiv(D_out, BLOCK_SIZE_D),
        triton.cdiv(H_out, BLOCK_SIZE_H),
        triton.cdiv(W_out, BLOCK_SIZE_W),
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out, groups,
        D, H, W,
        Kd, Kh, Kw,
        stride, stride, stride,
        padding, padding, padding,
        output_padding, output_padding, output_padding,
        1, 1, 1,  # output strides
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized 3D transposed convolution with custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register parameters as buffers to ensure they're preserved
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weight and bias (matching nn.ConvTranspose3d initialization)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.groups
        )