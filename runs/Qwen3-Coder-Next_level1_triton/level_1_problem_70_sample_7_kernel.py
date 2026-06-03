import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # [B, C_in, D, H, W]
    w_ptr,  # [C_in, C_out // groups, Kd, Kh, Kw]
    b_ptr,  # [C_out] or None
    out_ptr,  # [B, C_out, D_out, H_out, W_out]
    # Dimensions
    B, C_in, C_out, groups,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    dil_d, dil_h, dil_w,
    # Block sizes
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_Kd: tl.constexpr,
    BLOCK_Kh: tl.constexpr,
    BLOCK_Kw: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output channel range for this block
    c_out_start = pid_c_out * BLOCK_C_out
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_C_out)
    c_out_mask = c_out_offsets < C_out
    
    # Calculate output position
    d = pid_d
    h = pid_h
    w = pid_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_C_out,), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for c_in in range(0, C_in, BLOCK_C_in):
        c_in_offsets = c_in + tl.arange(0, BLOCK_C_in)
        c_in_mask = c_in_offsets < C_in
        
        # Calculate input position in the original space
        # For transposed conv: d_out = (d_in - 1) * stride - 2*pad + dil*(kd-1) + out_pad + 1
        # So d_in = (d_out - out_pad - 1 + 2*pad - dil*(kd-1)) / stride + 1
        # But we iterate over kernel positions instead
        
        # Iterate over kernel positions
        for kd in range(0, Kd, BLOCK_Kd):
            for kh in range(0, Kh, BLOCK_Kh):
                for kw in range(0, Kw, BLOCK_Kw):
                    kd_offsets = kd + tl.arange(0, BLOCK_Kd)
                    kh_offsets = kh + tl.arange(0, BLOCK_Kh)
                    kw_offsets = kw + tl.arange(0, BLOCK_Kw)
                    
                    # Calculate corresponding input positions
                    d_in = d - kd * dil_d + pad_d
                    h_in = h - kh * dil_h + pad_h
                    w_in = w - kw * dil_w + pad_w
                    
                    # Check if valid input position
                    valid_input = (
                        (d_in >= 0) & (d_in < D_in) &
                        (h_in >= 0) & (h_in < H_in) &
                        (w_in >= 0) & (w_in < W_in)
                    )
                    
                    # Calculate stride positions
                    d_in_strided = d_in // stride_d
                    h_in_strided = h_in // stride_h
                    w_in_strided = w_in // stride_w
                    
                    # Check if input position matches stride
                    valid_stride = (
                        (d_in % stride_d == 0) &
                        (h_in % stride_h == 0) &
                        (w_in % stride_w == 0)
                    )
                    
                    # Combined mask
                    valid = valid_input & valid_stride & c_in_mask[:, None, None, None] & c_out_mask[None, :, None, None]
                    
                    # Load input value
                    if valid_input:
                        x_offset = pid_b * (C_in * D_in * H_in * W_in) + \
                                  c_in_offsets[:, None, None, None] * (D_in * H_in * W_in) + \
                                  d_in * (H_in * W_in) + \
                                  h_in * W_in + \
                                  w_in
                        x_val = tl.load(x_ptr + x_offset, mask=valid_input & c_in_mask[:, None, None, None], other=0.0)
                    else:
                        x_val = tl.zeros((BLOCK_C_in, 1, 1, 1), dtype=tl.float32)
                    
                    # Load kernel weights
                    w_offset = c_in_offsets[:, None, None, None] * (C_out * Kd * Kh * Kw) + \
                              c_out_offsets[None, :, None, None] * (Kd * Kh * Kw) + \
                              (kd - pid_c_out * 0) * (Kh * Kw) +  # Simplified - groups not fully optimized yet
                              kh * Kw + kw
                    # Actually, need proper group handling - let's simplify for now
                    
                    # For simplicity, we'll handle groups in a separate way
                    # Group index for current output channel
                    group_idx = c_out_start // (C_out // groups)
                    c_in_group_start = group_idx * (C_in // groups)
                    c_in_group_offsets = c_in_group_start + (c_in_offsets % (C_in // groups))
                    
                    # Load kernel - simplified version without full group optimization
                    w_val = tl.zeros((BLOCK_C_in, BLOCK_C_out), dtype=tl.float32)
                    # This is complex - let's use a simpler approach
                    
    # Simplified implementation: process one output position at a time with inner loops
    # Reimplementing with cleaner approach
    pass


@triton.jit
def conv_transpose3d_kernel_v2(
    x_ptr,  # [B, C_in, D_in, H_in, W_in]
    w_ptr,  # [C_in, C_out // groups, Kd, Kh, Kw]
    b_ptr,  # [C_out] or None
    out_ptr,  # [B, C_out, D_out, H_out, W_out]
    # Dimensions
    B, C_in, C_out, groups,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    dil_d, dil_h, dil_w,
    # Block sizes
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_D_out: tl.constexpr,
    BLOCK_H_out: tl.constexpr,
    BLOCK_W_out: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output channel range for this block
    c_out_start = pid_c_out * BLOCK_C_out
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_C_out)
    c_out_mask = c_out_offsets < C_out
    
    # Calculate output position
    d = pid_d * BLOCK_D_out + tl.arange(0, BLOCK_D_out)
    h = pid_h * BLOCK_H_out + tl.arange(0, BLOCK_H_out)
    w = pid_w * BLOCK_W_out + tl.arange(0, BLOCK_W_out)
    
    # Create meshgrid for output positions
    d_grid = d[:, None, None]
    h_grid = h[None, :, None]
    w_grid = w[None, None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_D_out, BLOCK_H_out, BLOCK_W_out, BLOCK_C_out), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in in range(0, C_in, BLOCK_C_in):
        c_in_offsets = c_in + tl.arange(0, BLOCK_C_in)
        c_in_mask = c_in_offsets < C_in
        
        # Iterate over kernel dimensions
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Calculate corresponding input position for each output position
                    d_in = d_grid * stride_d - kd * dil_d + pad_d
                    h_in = h_grid * stride_h - kh * dil_h + pad_h
                    w_in = w_grid * stride_w - kw * dil_w + pad_w
                    
                    # Check if valid input position
                    valid_input = (
                        (d_in >= 0) & (d_in < D_in) &
                        (h_in >= 0) & (h_in < H_in) &
                        (w_in >= 0) & (w_in < W_in)
                    )
                    
                    # Load input values
                    x_offset = pid_b * (C_in * D_in * H_in * W_in) + \
                              c_in_offsets[None, None, None, :] * (D_in * H_in * W_in) + \
                              d_in[:, :, :, None] * (H_in * W_in) + \
                              h_in[:, :, :, None] * W_in + \
                              w_in[:, :, :, None]
                    x_val = tl.load(x_ptr + x_offset, mask=valid_input[:, :, :, None] & c_in_mask[None, None, None, :], other=0.0)
                    
                    # Load kernel weights
                    w_offset = c_in_offsets[None, None, None, :] * (C_out * Kd * Kh * Kw) + \
                              c_out_offsets[None, None, :, None] * (Kd * Kh * Kw) + \
                              kd * (Kh * Kw) + \
                              kh * Kw + \
                              kw
                    w_val = tl.load(w_ptr + w_offset, mask=c_in_mask[None, None, None, :] & c_out_mask[None, None, :, None], other=0.0)
                    
                    # Accumulate
                    acc += tl.sum(x_val * w_val[:, :, :, :, None], axis=3)
    
    # Add bias if present
    if b_ptr is not None:
        b = tl.load(b_ptr + c_out_offsets, mask=c_out_mask)
        acc += b[None, None, None, :]
    
    # Store result
    out_offset = pid_b * (C_out * D_out * H_out * W_out) + \
                c_out_offsets[None, None, None, :] * (D_out * H_out * W_out) + \
                d_grid[:, :, :, None] * (H_out * W_out) + \
                h_grid[:, :, :, None] * W_out + \
                w_grid[:, :, :, None]
    
    tl.store(out_ptr + out_offset, acc, mask=valid_input[:, :, :, None] & c_out_mask[None, None, None, :])


@triton.jit
def conv_transpose3d_kernel_v3(
    x_ptr,  # [B, C_in, D_in, H_in, W_in]
    w_ptr,  # [C_in, C_out // groups, Kd, Kh, Kw]
    b_ptr,  # [C_out] or None
    out_ptr,  # [B, C_out, D_out, H_out, W_out]
    # Dimensions
    B, C_in, C_out, groups,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    dil_d, dil_h, dil_w,
    # Block sizes
    BLOCK_C_out: tl.constexpr,
    BLOCK_C_in: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output channel range
    c_out_start = pid_c_out * BLOCK_C_out
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_C_out)
    c_out_mask = c_out_offsets < C_out
    
    # Calculate output position
    d = pid_d
    h = pid_h
    w = pid_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_C_out,), dtype=tl.float32)
    
    # Iterate over input channels in blocks
    for c_in in range(0, C_in, BLOCK_C_in):
        c_in_offsets = c_in + tl.arange(0, BLOCK_C_in)
        c_in_mask = c_in_offsets < C_in
        
        # Iterate over kernel dimensions
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Calculate corresponding input position
                    d_in = d * stride_d - kd * dil_d + pad_d
                    h_in = h * stride_h - kh * dil_h + pad_h
                    w_in = w * stride_w - kw * dil_w + pad_w
                    
                    # Check if valid input position
                    valid_input = (
                        (d_in >= 0) & (d_in < D_in) &
                        (h_in >= 0) & (h_in < H_in) &
                        (w_in >= 0) & (w_in < W_in)
                    )
                    
                    if valid_input:
                        # Load input value
                        x_offset = pid_b * (C_in * D_in * H_in * W_in) + \
                                  c_in_offsets * (D_in * H_in * W_in) + \
                                  d_in * (H_in * W_in) + \
                                  h_in * W_in + \
                                  w_in
                        x_val = tl.load(x_ptr + x_offset, mask=c_in_mask, other=0.0)
                        
                        # Load kernel weights
                        # For transposed conv: weight shape is [C_in, C_out // groups, Kd, Kh, Kw]
                        # But we need to handle groups properly
                        group_size_c_in = C_in // groups
                        group_size_c_out = C_out // groups
                        group_idx = c_out_start // group_size_c_out
                        c_in_group_start = group_idx * group_size_c_in
                        c_in_group = c_in_offsets - c_in_group_start
                        
                        w_offset = c_in_group * (C_out * Kd * Kh * Kw) + \
                                  (c_out_offsets - c_out_start) * (Kd * Kh * Kw) + \
                                  kd * (Kh * Kw) + \
                                  kh * Kw + \
                                  kw
                        w_val = tl.load(w_ptr + w_offset, mask=c_in_mask & c_out_mask, other=0.0)
                        
                        # Accumulate
                        acc += x_val[:, None] * w_val[None, :]
    
    # Add bias if present
    if b_ptr is not None:
        b = tl.load(b_ptr + c_out_offsets, mask=c_out_mask)
        acc += b
    
    # Store result
    out_offset = pid_b * (C_out * D_out * H_out * W_out) + \
                c_out_offsets * (D_out * H_out * W_out) + \
                d * (H_out * W_out) + \
                h * W_out + \
                w
    tl.store(out_ptr + out_offset, acc, mask=c_out_mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out_group, Kd, Kh, Kw = weight.shape
    C_out = C_in_w * C_out_group  # Should equal weight.shape[0] * weight.shape[1]
    
    # Calculate output dimensions manually
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (Kd - 1) + output_padding[0] + 1
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (Kh - 1) + output_padding[1] + 1
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + dilation[2] * (Kw - 1) + output_padding[2] + 1
    
    # Create output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Define block sizes
    BLOCK_C_out = 8
    BLOCK_C_in = 16
    
    # Calculate grid dimensions
    grid = lambda meta: (
        B,
        triton.cdiv(C_out, meta['BLOCK_C_out']),
        D_out,
        H_out,
        W_out,
    )
    
    # Launch kernel
    conv_transpose3d_kernel_v3[grid](
        x, weight, bias, out,
        B, C_in, C_out, groups,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        Kd, Kh, Kw,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        dilation[0], dilation[1], dilation[2],
        BLOCK_C_out=BLOCK_C_out,
        BLOCK_C_in=BLOCK_C_in,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of each dimension in the output shape. 
                                                  Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding,
            self.dilation, self.groups
        )