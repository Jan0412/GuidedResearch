import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, H, W,
    C_out, KH, KW,
    S, PH, PW, DH, DW,
    OH, OW,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_oc, w_stride_ic, w_stride_kh, w_stride_kw,
    out_stride_b, out_stride_oc, out_stride_oh, out_stride_ow,
    BLOCK_SIZE_OC: tl.constexpr,
):
    # program_id(0) maps to (batch, oh, ow)
    pid_0 = tl.program_id(0)
    # program_id(1) maps to block of output channels
    pid_oc = tl.program_id(1)

    # Decode pid_0 into b, oh, ow
    # OH and OW are the output dimensions
    ow = pid_0 % OW
    oh = (pid_0 // OW) % OH
    b = pid_0 // (OW * OH)

    # Output channel offsets for this block
    oc_offsets = pid_oc * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)
    oc_mask = oc_offsets < C_out

    # Accumulator for the dot product
    acc = tl.zeros([BLOCK_SIZE_OC], dtype=tl.float32)

    # Loop over input channels and kernel dimensions
    # For small kernels, simple loops are efficient in Triton
    for ic in range(C_in):
        for kh in range(KH):
            for kw in range(KW):
                # Calculate input coordinates
                ih = oh * S - PH + kh * DH
                iw = ow * S - PW + kw * DW

                # Boundary check for padding
                if ih >= 0 and ih < H and iw >= 0 and iw < W:
                    # Load single value from input x
                    x_val = tl.load(x_ptr + b * x_stride_b + ic * x_stride_c + ih * x_stride_h + iw * x_stride_w)
                    
                    # Load vector of weights for the current (ic, kh, kw) across BLOCK_SIZE_OC
                    w_ptr_base = w_ptr + oc_offsets * w_stride_oc + ic * w_stride_ic + kh * w_stride_kh + kw * w_stride_kw
                    w_vec = tl.load(w_ptr_base, mask=oc_mask, other=0.0)
                    
                    acc += x_val * w_vec

    # Add bias if available
    if b_ptr is not None:
        bias_vec = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0)
        acc += bias_vec

    # Store the result
    out_ptr_base = out_ptr + b * out_stride_b + oc_offsets * out_stride_oc + oh * out_stride_oh + ow * out_stride_ow
    tl.store(out_ptr_base, acc, mask=oc_mask)


def triton_conv2d(x, weight, bias, stride, padding, dilation):
    # Input shapes
    B, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    S = stride
    PH, PW = padding
    DH, DW = dilation

    # Calculate output dimensions
    OH = (H + 2 * PH - DH * (KH - 1) - 1) // S + 1
    OW = (W + 2 * PW - DW * (KW - 1) - 1) // S + 1

    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    out = torch.empty((B, C_out, OH, OW), device=x.device, dtype=x.dtype)

    # Get strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
    w_stride_oc, w_stride_ic, w_stride_kh, w_stride_kw = weight.stride()
    out_stride_b, out_stride_oc, out_stride_oh, out_stride_ow = out.stride()

    BLOCK_SIZE_OC = 32
    # Grid: (batch * oh * ow, ceil(C_out / BLOCK_SIZE_OC))
    grid = (B * OH * OW, triton.cdiv(C_out, BLOCK_SIZE_OC))

    conv2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, H, W,
        C_out, KH, KW,
        S, PH, PW, DH, DW,
        OH, OW,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        w_stride_oc, w_stride_ic, w_stride_kh, w_stride_kw,
        out_stride_b, out_stride_oc, out_stride_oh, out_stride_ow,
        BLOCK_SIZE_OC=BLOCK_SIZE_OC,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        # We use nn.Conv2d to manage parameters, but we will call our custom kernel in forward
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract weights and bias from the nn.Conv2d module
        weight = self.conv2d.weight
        bias = self.conv2d.bias if self.conv2d.bias is not None else None
        
        return triton_conv2d(
            x, 
            weight, 
            bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )