import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input tensor: (B, C_in, D, H, W)
    w_ptr,  # Weight tensor: (C_in, C_out // groups, kD, kH, kW)
    bias_ptr,  # Bias tensor: (C_out,) or None
    out_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    B, C_in, C_out, groups,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    dil_d, dil_h, dil_w,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Compute output position
    out_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for valid output indices
    mask_d = out_d < D_out
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    mask_out = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels and groups
    c_out_group = pid_c_out * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    mask_c_out = c_out_group < C_out
    
    for c_in_idx in range(0, C_in, BLOCK_SIZE_C_IN):
        c_in = c_in_idx + tl.arange(0, BLOCK_SIZE_C_IN)
        mask_c_in = c_in < C_in
        
        # Compute which group this channel belongs to
        group_idx = c_in // (C_in // groups)
        
        # Only process if c_out is in the same group as c_in
        c_out_valid = (c_out_group % (C_out // groups)) == (c_in % (C_out // groups))
        mask_group = c_out_valid[:, None, None] if BLOCK_SIZE_C_OUT > 1 else c_out_valid
        
        # Broadcast masks
        mask_c = mask_c_out[:, None, None] & mask_c_in[None, :, None]
        
        # Process kernel elements
        for kd in range(0, kD, BLOCK_SIZE_KD):
            kernel_d = kd + tl.arange(0, BLOCK_SIZE_KD)
            mask_kd = kernel_d < kD
            
            for kh in range(0, kH, BLOCK_SIZE_KH):
                kernel_h = kh + tl.arange(0, BLOCK_SIZE_KH)
                mask_kh = kernel_h < kH
                
                for kw in range(0, kW, BLOCK_SIZE_KW):
                    kernel_w = kw + tl.arange(0, BLOCK_SIZE_KW)
                    mask_kw = kernel_w < kW
                    
                    # Compute input positions for this kernel element
                    # For transposed conv: input_pos = (output_pos - (kernel_pos-1) + pad - output_pad) // stride
                    in_d = (out_d[:, None, None] - (kernel_d[None, :, None] - 1) + pad_d - output_pad_d) // stride_d
                    in_h = (out_h[None, :, None] - (kernel_h[None, None, :] - 1) + pad_h - output_pad_w) // stride_h
                    in_w = (out_w[None, None, :] - (kernel_w[None, None, :] - 1) + pad_w - output_pad_w) // stride_w
                    
                    # Check if input positions are valid
                    valid_in = (in_d >= 0) & (in_d < D_in) & (in_h >= 0) & (in_h < H_in) & (in_w >= 0) & (in_w < W_in)
                    mask_valid = valid_in & mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
                    
                    # Compute input pointers
                    in_batch_offset = pid_b * (C_in * D_in * H_in * W_in)
                    in_c_offset = c_in[None, :, None] * (D_in * H_in * W_in)
                    in_d_offset = in_d * (H_in * W_in)
                    in_h_offset = in_h * W_in
                    in_w_offset = in_w
                    
                    # Load input values
                    in_ptr = x_ptr + in_batch_offset + in_c_offset + in_d_offset + in_h_offset + in_w_offset
                    in_vals = tl.load(in_ptr, mask=mask_valid & mask_c, other=0.0)
                    
                    # Load weight values
                    w_ptr_offset = (c_in[None, :, None] * (C_out * kD * kH * kW) + 
                                   c_out_group[:, None, None] * (kD * kH * kW) +
                                   kernel_d[None, :, None] * (kH * kW) +
                                   kernel_h[None, None, :] * kW +
                                   kernel_w[None, None, :])
                    w_vals = tl.load(w_ptr + w_ptr_offset, mask=mask_c & mask_kd[None, :, None] & mask_kh[None, None, :] & mask_kw[None, None, :], other=0.0)
                    
                    # Accumulate: in_vals has shape (BLOCK_SIZE_D, BLOCK_SIZE_C_IN, BLOCK_SIZE_W)
                    # w_vals has shape (BLOCK_SIZE_C_OUT, BLOCK_SIZE_C_IN, BLOCK_SIZE_KD, BLOCK_SIZE_KH, BLOCK_SIZE_KW)
                    # We need to compute: out[d,h,w] += sum_c in[d',h',w'] * w[c_in, c_out, kd, kh, kw]
                    # where d' = (d - (kd-1) + pad - output_pad) // stride, etc.
                    
                    # Reshape for broadcasting: (BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C_IN, BLOCK_SIZE_KD, BLOCK_SIZE_KH, BLOCK_SIZE_KW)
                    in_broadcast = in_vals[:, None, :, None, None, None, None]  # (BD, 1, BCIN, 1, 1, 1, 1)
                    w_broadcast = w_vals[:, :, None, None, None, None, None]  # (BCOUT, BCIN, 1, 1, 1, 1, 1)
                    
                    # Multiply and accumulate
                    if BLOCK_SIZE_C_IN > 1:
                        acc += tl.sum(in_broadcast * w_broadcast, axis=2)  # Sum over C_IN dimension
                    else:
                        acc += in_broadcast[:, :, :, 0, 0, 0, 0] * w_vals[:, 0, 0, 0, 0]
    
    # Apply bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + c_out_group, mask=mask_c_out, other=0.0)
        acc += bias[:, None, None, None]
    
    # Store output
    out_batch_offset = pid_b * (C_out * D_out * H_out * W_out)
    out_c_offset = c_out_group[:, None, None] * (D_out * H_out * W_out)
    out_d_offset = out_d[None, :, None] * (H_out * W_out)
    out_h_offset = out_h[None, None, :] * W_out
    out_w_offset = out_w[None, None, :]
    
    out_ptr_final = out_ptr + out_batch_offset + out_c_offset + out_d_offset + out_h_offset + out_w_offset
    tl.store(out_ptr_final, acc, mask=mask_c_out[:, None, None] & mask_out)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    """
    Triton implementation of ConvTranspose3d forward pass.
    
    Args:
        x: Input tensor of shape (B, C_in, D, H, W)
        weight: Weight tensor of shape (C_in, C_out // groups, kD, kH, kW)
        bias: Optional bias tensor of shape (C_out,)
        stride, padding, output_padding, dilation, groups: ConvTranspose3d parameters
        
    Returns:
        Output tensor of shape (B, C_out, D_out, H_out, W_out)
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D_in, H_in, W_in = x.shape
    C_in_, C_out_g, kD, kH, kW = weight.shape
    C_out = C_out_g * groups
    
    # Handle stride, padding, dilation, output_padding
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_d = pad_h = pad_w = padding
    else:
        pad_d, pad_h, pad_w = padding
        
    if isinstance(output_padding, int):
        output_pad_d = output_pad_h = output_pad_w = output_padding
    else:
        output_pad_d, output_pad_h, output_pad_w = output_padding
        
    if isinstance(dilation, int):
        dil_d = dil_h = dil_w = dilation
    else:
        dil_d, dil_h, dil_w = dilation
    
    # Compute output dimensions
    D_out = (D_in - 1) * stride_d - 2 * pad_d + dil_d * (kD - 1) + output_pad_d + 1
    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + output_pad_h + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + output_pad_w + 1
    
    # Prepare output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Define block sizes for optimization
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_C_OUT = min(16, C_out)
    BLOCK_SIZE_D = min(4, D_out)
    BLOCK_SIZE_H = min(4, H_out)
    BLOCK_SIZE_W = min(4, W_out)
    BLOCK_SIZE_C_IN = min(16, C_in)
    BLOCK_SIZE_KD = min(3, kD)
    BLOCK_SIZE_KH = min(3, kH)
    BLOCK_SIZE_KW = min(3, kW)
    
    # Compute grid dimensions
    grid = lambda meta: (
        B,
        (C_out + meta["BLOCK_SIZE_C_OUT"] - 1) // meta["BLOCK_SIZE_C_OUT"],
        (D_out + meta["BLOCK_SIZE_D"] - 1) // meta["BLOCK_SIZE_D"],
        (H_out + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (W_out + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, groups,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        output_pad_d, output_pad_h, output_pad_w,
        dil_d, dil_h, dil_w,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        BLOCK_SIZE_KD=BLOCK_SIZE_KD,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for ConvTranspose3d.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                                                stride=stride, padding=padding, output_padding=output_padding, 
                                                dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Uses custom Triton kernel for the transposed 3D convolution.
        """
        return triton_conv_transpose3d(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias,
            stride=self.conv_transpose3d.stride,
            padding=self.conv_transpose3d.padding,
            output_padding=self.conv_transpose3d.output_padding,
            dilation=self.conv_transpose3d.dilation,
            groups=self.conv_transpose3d.groups,
        )