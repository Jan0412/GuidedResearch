import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # [B, C_in, D, H, W]
    w_ptr,  # [C_in, C_out // G, Kd, Kh, Kw] (transposed conv weight layout)
    b_ptr,  # [C_out] or None
    out_ptr,  # [B, C_out, D_out, H_out, W_out]
    # Dimensions
    B, C_in, C_out, G,  # batch, channels, groups
    D, H, W,  # input dims
    D_out, H_out, W_out,  # output dims
    Kd, Kh, Kw,  # kernel dims
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    dil_d, dil_h, dil_w,
    # Strides
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
    stride_ouc, stride_oud, stride_ouh, stride_ouw,
    # Block sizes
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program IDs for output tensor
    pid_c_out = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    pid_b = tl.program_id(4)
    
    # Compute output channel range
    c_out_start = pid_c_out * BLOCK_C_OUT
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_C_OUT)
    c_out_mask = c_out_offsets < C_out
    
    # Compute output position
    d_out = pid_d
    h_out = pid_h
    w_out = pid_w
    
    # Compute corresponding input position (for transposed conv)
    # d_in = (d_out - Kd + 1 + pad_d + out_pad_d) // stride_d
    # But for transposed conv, we have: d_out = (d_in - 1) * stride_d - 2*pad_d + (Kd - 1)*dil_d + 1 + out_pad_d
    # So: d_in = ((d_out - 1 - out_pad_d + 2*pad_d) // stride_d) + 1
    # Simplified: d_in = (d_out - out_pad_d - 1) // stride_d + 1 (when stride_d > 0)
    
    # For transposed convolution: each output element is sum over input elements and kernel
    # out[b, c_out, d_out, h_out, w_out] = sum_{c_in, kd, kh, kw} x[b, c_in, d_in, h_in, w_in] * w[c_in, c_out, kd, kh, kw] + b[c_out]
    # where d_in = d_out - (Kd-1)*dil_d + kd*dil_d - pad_d + out_pad_d, similarly for h_in, w_in
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_C_OUT,), dtype=tl.float32)
    
    # Iterate over input channels
    for c_in in range(C_in // BLOCK_C_IN):
        c_in_start = c_in * BLOCK_C_IN
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_C_IN)
        
        # Compute input coordinates for this output position
        # For transposed conv: d_in = (d_out - out_pad_d - 1 + pad_d - kd*dil_d) // stride_d + 1
        # But more intuitively: d_out = (d_in - 1)*stride_d + (kd-1)*dil_d + 1 - 2*pad_d + out_pad_d
        
        # Let's compute d_in such that when kernel is applied at d_in, it contributes to d_out
        # d_out = (d_in - 1)*stride_d + (kd-1)*dil_d + 1 - 2*pad_d + out_pad_d
        # => d_in = (d_out - 1 + 2*pad_d - out_pad_d - (kd-1)*dil_d) // stride_d + 1
        
        # Actually, for transposed conv with stride s: the input is upsampled by inserting s-1 zeros between elements
        # Then normal conv is applied. So effectively:
        # d_out = d_in * stride_d + kd * dil_d - pad_d + out_pad_d (approximately)
        
        # Let me reconsider: standard formula for transposed conv output is:
        # D_out = (D_in - 1) * stride_d - 2 * pad_d + (Kd - 1) * dil_d + 1 + out_pad_d
        
        # For each output position (d_out), we need to find which input positions contribute:
        # d_in must satisfy: d_out = d_in * stride_d + kd * dil_d - pad_d + out_pad_d (simplified)
        # => d_in = (d_out + pad_d - out_pad_d - kd * dil_d) / stride_d
        
        # Let's use the correct formula:
        # In transposed conv, kernel is applied as if input was upsampled with stride-1 spacing
        # d_out = d_in * stride_d + (kd - 1) * dil_d - pad_d + out_pad_d
        
        # So for fixed d_out and kd, we get: d_in = (d_out - (kd - 1) * dil_d + pad_d - out_pad_d) / stride_d
        
        kd_range = tl.arange(0, Kd)
        kh_range = tl.arange(0, Kh)
        kw_range = tl.arange(0, Kw)
        
        # Compute d_in for all kernel positions
        d_in = (d_out - (kd_range - 1) * dil_d + pad_d - out_pad_d) // stride_d
        
        # Check if d_in is valid
        d_valid = (d_in >= 0) & (d_in < D)
        
        # Similarly for h_in and w_in
        h_in = (h_out - (kh_range - 1) * dil_h + pad_h - out_pad_h) // stride_h
        h_valid = (h_in >= 0) & (h_in < H)
        
        w_in = (w_out - (kw_range - 1) * dil_w + pad_w - out_pad_w) // stride_w
        w_valid = (w_in >= 0) & (w_in < W)
        
        # Load input values where valid
        if d_valid and h_valid and w_valid:
            # For simplicity, handle one kernel position at a time
            for kd in range(Kd):
                for kh in range(Kh):
                    for kw in range(Kw):
                        # Compute input position
                        d_in_k = (d_out - (kd - 1) * dil_d + pad_d - out_pad_d) // stride_d
                        h_in_k = (h_out - (kh - 1) * dil_h + pad_h - out_pad_h) // stride_h
                        w_in_k = (w_out - (kw - 1) * dil_w + pad_w - out_pad_w) // stride_w
                        
                        # Check bounds
                        if (d_in_k >= 0 and d_in_k < D and 
                            h_in_k >= 0 and h_in_k < H and 
                            w_in_k >= 0 and w_in_k < W):
                            
                            # Load input: x[b, c_in, d_in_k, h_in_k, w_in_k]
                            x_offset = pid_b * stride_xn + c_in * stride_xc + d_in_k * stride_xd + h_in_k * stride_xh + w_in_k * stride_xw
                            x_val = tl.load(x_ptr + x_offset, mask=c_in_mask, other=0.0)
                            
                            # Load weight: w[c_in, c_out, kd, kh, kw]
                            w_offset = c_in * stride_wn + c_out_offsets * stride_wc + kd * stride_wd + kh * stride_wh + kw * stride_ww
                            w_val = tl.load(w_ptr + w_offset, mask=c_out_mask, other=0.0)
                            
                            # Accumulate
                            acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
        acc += bias
    
    # Store result
    out_offset = pid_b * B * stride_ouc + c_out_offsets * stride_ouc + pid_d * stride_oud + pid_h * stride_ouh + pid_w * stride_ouw
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty), mask=c_out_mask)


def triton_conv_transpose3d(x, weight, bias, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
    """
    Custom Triton implementation of ConvTranspose3d
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D, H, W = x.shape
    C_in2, C_out_per_group, Kd, Kh, Kw = weight.shape
    C_out = C_in2 * C_out_per_group  # should equal C_in for normal conv, but here weight shape is [C_in, C_out//G, Kd, Kh, Kw]
    
    # Validate
    assert C_in == C_in2, f"Input channels mismatch: {C_in} vs {C_in2}"
    
    # Compute output dimensions
    D_out = (D - 1) * stride - 2 * padding + (Kd - 1) * dilation + 1 + output_padding
    H_out = (H - 1) * stride - 2 * padding + (Kh - 1) * dilation + 1 + output_padding
    W_out = (W - 1) * stride - 2 * padding + (Kw - 1) * dilation + 1 + output_padding
    
    # Create output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Strides for inputs
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw = x.stride()
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww = weight.stride()
    stride_ouc, stride_oud, stride_ouh, stride_ouw = out.stride()
    
    # Determine block sizes
    BLOCK_C_OUT = min(32, C_out)
    BLOCK_C_IN = min(16, C_in)
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4
    
    # Grid dimensions
    grid = lambda meta: (
        (C_out + meta['BLOCK_C_OUT'] - 1) // meta['BLOCK_C_OUT'],
        D_out,
        H_out,
        W_out,
        B
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, groups,
        D, H, W,
        D_out, H_out, W_out,
        Kd, Kh, Kw,
        stride, stride, stride,
        padding, padding, padding,
        output_padding, output_padding, output_padding,
        dilation, dilation, dilation,
        stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
        stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
        stride_ouc, stride_oud, stride_ouh, stride_ouw,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_C_IN=BLOCK_C_IN,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize with same parameters as original
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias parameters (same as nn.ConvTranspose3d)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )