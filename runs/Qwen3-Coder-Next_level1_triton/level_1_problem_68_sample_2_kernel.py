import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # [B, Cin, D, H, W]
    w_ptr,  # [Cin, Cout // groups, Kd, Kh, Kw]
    bias_ptr,  # [Cout] or None
    out_ptr,  # [B, Cout, D_out, H_out, W_out]
    # Dimensions
    B, Cin, Cout, groups,
    D, H, W,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    # Strides for each tensor
    stride_x_batch, stride_x_ch, stride_x_d, stride_x_h, stride_x_w,
    stride_w_in_ch, stride_w_out_ch, stride_w_kd, stride_w_kh, stride_w_kw,
    stride_out_batch, stride_out_ch, stride_out_d, stride_out_h, stride_out_w,
    # Block sizes
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output position
    out_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Output mask
    mask_d = out_d < D
    mask_h = out_h < H
    mask_w = out_w < W
    mask_ohw = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    
    # Output channel indices
    cout_start = pid_cout * BLOCK_SIZE_COUT
    cout_ids = cout_start + tl.arange(0, BLOCK_SIZE_COUT)
    mask_cout = cout_ids < Cout
    
    # Calculate input position (using transposed convolution relation)
    # For transposed conv: out_d = stride_d * (in_d - 1) + (Kd - 1 - pad_d) - out_pad_d + d
    # Solve for in_d: in_d = (out_d - (Kd - 1 - pad_d) + out_pad_d - d) / stride_d
    # where d is the kernel index (0..Kd-1)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels
    for c_in_start in range(0, Cin, BLOCK_SIZE_CIN):
        c_in_ids = c_in_start + tl.arange(0, BLOCK_SIZE_CIN)
        mask_cin = c_in_ids < Cin
        
        # Loop over kernel depth
        for kd in range(0, Kd, BLOCK_SIZE_KD):
            kd_start = kd
            kd_ids = kd_start + tl.arange(0, BLOCK_SIZE_KD)
            mask_kd = kd_ids < Kd
            
            # Loop over kernel height
            for kh in range(0, Kh, BLOCK_SIZE_KH):
                kh_start = kh
                kh_ids = kh_start + tl.arange(0, BLOCK_SIZE_KH)
                mask_kh = kh_ids < Kh
                
                # Loop over kernel width
                for kw in range(0, Kw, BLOCK_SIZE_KW):
                    kw_start = kw
                    kw_ids = kw_start + tl.arange(0, BLOCK_SIZE_KW)
                    mask_kw = kw_ids < Kw
                    
                    # Calculate input indices for this kernel position
                    # For each output position, we need to check which input positions contribute
                    # out_d = stride_d * in_d - stride_d + Kd - 1 - pad_d + kd
                    # => in_d = (out_d + stride_d - Kd + 1 + pad_d - kd) / stride_d
                    
                    in_d = (out_d[:, None, None] + stride_d - Kd + 1 + pad_d - kd_ids[None, :, None]) // stride_d
                    in_h = (out_h[None, :, None] + stride_h - Kh + 1 + pad_h - kh_ids[None, None, :]) // stride_h
                    in_w = (out_w[None, None, :] + stride_w - Kw + 1 + pad_w - kw_ids[None, None, None]) // stride_w
                    
                    # Check if input indices are valid
                    valid_d = (in_d >= 0) & (in_d < D)
                    valid_h = (in_h >= 0) & (in_h < H)
                    valid_w = (in_w >= 0) & (in_w < W)
                    valid = valid_d & valid_h & valid_w
                    
                    # Load input values where valid
                    x_offset = (
                        pid_b * stride_x_batch +
                        c_in_ids[None, :, None, None, None] * stride_x_ch +
                        in_d[None, :, :, None, None] * stride_x_d +
                        in_h[None, :, None, :, None] * stride_x_h +
                        in_w[None, :, None, None, :] * stride_x_w
                    )
                    
                    # Load weights
                    w_offset = (
                        c_in_ids[:, None, None, None, None] * stride_w_in_ch +
                        pid_cout * BLOCK_SIZE_COUT * stride_w_out_ch +
                        (kd_ids[None, :, None, None, None]) * stride_w_kd +
                        (kh_ids[None, None, :, None, None]) * stride_w_kh +
                        (kw_ids[None, None, None, :, None]) * stride_w_kw
                    )
                    
                    # Actually, we need to restructure for proper indexing
                    # Let's do a more efficient approach: compute contributions per kernel position
                    
                    # For each kernel position, compute contribution to output
                    for ki_d in range(BLOCK_SIZE_KD):
                        if kd + ki_d >= Kd: continue
                        actual_kd = kd + ki_d
                        for ki_h in range(BLOCK_SIZE_KH):
                            if kh + ki_h >= Kh: continue
                            actual_kh = kh + ki_h
                            for ki_w in range(BLOCK_SIZE_KW):
                                if kw + ki_w >= Kw: continue
                                actual_kw = kw + ki_w
                                
                                # Calculate input position that contributes to output
                                in_d_idx = (out_d[:, None, None] - actual_kd + pad_d - out_pad_d) // stride_d + 1
                                in_h_idx = (out_h[None, :, None] - actual_kh + pad_h - out_pad_h) // stride_h + 1
                                in_w_idx = (out_w[None, None, :] - actual_kw + pad_w - out_pad_w) // stride_w + 1
                                
                                # Check validity
                                valid_mask = (
                                    (in_d_idx >= 0) & (in_d_idx < D) &
                                    (in_h_idx >= 0) & (in_h_idx < H) &
                                    (in_w_idx >= 0) & (in_w_idx < W)
                                )
                                
                                # Load input values
                                x_vals = tl.load(
                                    x_ptr + pid_b * stride_x_batch + 
                                    c_in_ids[None, :, None, None, None] * stride_x_ch +
                                    in_d_idx[None, :, :, None, None] * stride_x_d +
                                    in_h_idx[None, :, None, :, None] * stride_x_h +
                                    in_w_idx[None, :, None, None, :] * stride_x_w,
                                    mask=valid_mask[None, :, :, None, None],
                                    other=0.0
                                )
                                
                                # Load weight values
                                w_vals = tl.load(
                                    w_ptr + c_in_ids[:, None, None, None, None] * stride_w_in_ch +
                                    pid_cout * BLOCK_SIZE_COUT * stride_w_out_ch +
                                    actual_kd * stride_w_kd +
                                    actual_kh * stride_w_kh +
                                    actual_kw * stride_w_kw,
                                    mask=mask_cin[:, None, None, None, None]
                                )
                                
                                # Compute contribution
                                contrib = tl.sum(x_vals * w_vals[None, :, :, :, :], axis=1)
                                acc += contrib
    
    # Store result
    out_offset = (
        pid_b * stride_out_batch +
        cout_ids[None, :, None, None, None] * stride_out_ch +
        out_d[None, None, :, None, None] * stride_out_d +
        out_h[None, None, None, :, None] * stride_out_h +
        out_w[None, None, None, None, :] * stride_out_w
    )
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + cout_ids, mask=mask_cout)
        acc += bias[None, :, None, None, None]
    
    tl.store(
        out_ptr + out_offset,
        acc,
        mask=mask_cout[None, :, None, None, None] & mask_ohw[None, None, :, :, :]
    )


def triton_conv_transpose3d(x, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), 
                           output_padding=(0, 0, 0), groups=1):
    """
    Triton implementation of ConvTranspose3d forward pass.
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, Cin, D, H, W = x.shape
    Cin2, Cout_per_group, Kd, Kh, Kw = weight.shape
    assert Cin == Cin2, f"Input channels mismatch: {Cin} vs {Cin2}"
    
    # Output dimensions
    D_out = (D - 1) * stride[0] - 2 * padding[0] + (Kd - 1 - output_padding[0]) + 1
    H_out = (H - 1) * stride[1] - 2 * padding[1] + (Kh - 1 - output_padding[1]) + 1
    W_out = (W - 1) * stride[2] - 2 * padding[2] + (Kw - 1 - output_padding[2]) + 1
    
    # Create output tensor
    out = torch.empty(B, Cout, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Calculate strides
    stride_x_batch = x.stride(0)
    stride_x_ch = x.stride(1)
    stride_x_d = x.stride(2)
    stride_x_h = x.stride(3)
    stride_x_w = x.stride(4)
    
    stride_w_in_ch = weight.stride(0)
    stride_w_out_ch = weight.stride(1)
    stride_w_kd = weight.stride(2)
    stride_w_kh = weight.stride(3)
    stride_w_kw = weight.stride(4)
    
    stride_out_batch = out.stride(0)
    stride_out_ch = out.stride(1)
    stride_out_d = out.stride(2)
    stride_out_h = out.stride(3)
    stride_out_w = out.stride(4)
    
    # Grid dimensions
    grid = lambda meta: (
        B,
        triton.cdiv(Cout, meta["BLOCK_SIZE_COUT"]),
        triton.cdiv(D_out, meta["BLOCK_SIZE_D"]),
        triton.cdiv(H_out, meta["BLOCK_SIZE_H"]),
        triton.cdiv(W_out, meta["BLOCK_SIZE_W"]),
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, Cin, Cout, groups,
        D, H, W,
        Kd, Kh, Kw,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        stride_x_batch, stride_x_ch, stride_x_d, stride_x_h, stride_x_w,
        stride_w_in_ch, stride_w_out_ch, stride_w_kd, stride_w_kh, stride_w_kw,
        stride_out_batch, stride_out_ch, stride_out_d, stride_out_h, stride_out_w,
        BLOCK_SIZE_CIN=32,
        BLOCK_SIZE_COUT=32,
        BLOCK_SIZE_D=8,
        BLOCK_SIZE_H=8,
        BLOCK_SIZE_W=8,
        BLOCK_SIZE_KD=3,
        BLOCK_SIZE_KH=3,
        BLOCK_SIZE_KW=3,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        
        # Initialize the conv layer parameters
        kernel_depth, kernel_width, kernel_height = kernel_size
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding, 
            groups=groups, bias=bias
        )
        # We'll use this to get the weights initially, then we can replace with custom kernel
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the trained weights from the original layer
        return triton_conv_transpose3d(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias if self.use_bias else None,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )