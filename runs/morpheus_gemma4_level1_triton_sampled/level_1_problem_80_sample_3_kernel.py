import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W,
    C_out, H_out, W_out,
    stride, padH, padW, dilH, dilW,
    has_bias: tl.constexpr,
    kH: tl.constexpr, kW: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
):
    # Grid: (N, C_out, H_out, W_out)
    n = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = 0.0
    
    # Loop over input channels in blocks
    for ic_start in range(0, C_in, BLOCK_C_IN):
        ic_offsets = ic_start + tl.arange(0, BLOCK_C_IN)
        mask_ic = ic_offsets < C_in

        # Loop over kernel height and width (constexpr)
        for kh in range(kH):
            for kw in range(kW):
                # Calculate input coordinates
                ih = oh * stride - padH + kh * dilH
                iw = ow * stride - padW + kw * dilW

                # Boundary check for padding
                if ih >= 0 and ih < H and iw >= 0 and iw < W:
                    # x shape: (N, C_in, H, W)
                    # Offset: n * (C_in * H * W) + ic * (H * W) + ih * W + iw
                    x_off = n * (C_in * H * W) + ic_offsets * (H * W) + ih * W + iw
                    
                    # w shape: (C_out, C_in, kH, kW)
                    # Offset: oc * (C_in * kH * kW) + ic * (kH * kW) + kh * kW + kw
                    w_off = oc * (C_in * kH * kW) + ic_offsets * (kH * kW) + kh * kW + kw

                    x_val = tl.load(x_ptr + x_off, mask=mask_ic, other=0.0)
                    w_val = tl.load(w_ptr + w_off, mask=mask_ic, other=0.0)
                    acc += tl.sum(x_val * w_val)

    if has_bias:
        acc += tl.load(b_ptr + oc)

    # Store result in out shape: (N, C_out, H_out, W_out)
    out_off = n * (C_out * H_out * W_out) + oc * (H_out * W_out) + oh * W_out + ow
    tl.store(out_ptr + out_off, acc)

def triton_conv2d(x, weight, bias, stride, padding, dilation, kernel_size):
    # Input dimensions
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    padH, padW = padding
    dilH, dilW = dilation

    # Calculate output dimensions
    H_out = (H + 2 * padH - dilH * (kH - 1) - 1) // stride + 1
    W_out = (W + 2 * padW - dilW * (kW - 1) - 1) // stride + 1

    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    bias_ptr = bias if bias is not None else torch.zeros(1, device=x.device, dtype=x.dtype)
    if bias is not None:
        bias_ptr = bias.contiguous()

    grid = (N, C_out, H_out, W_out)
    
    conv2d_kernel[grid](
        x, weight, bias_ptr, out,
        N, C_in, H, W,
        C_out, H_out, W_out,
        stride, padH, padW, dilH, dilW,
        has_bias=1 if bias is not None else 0,
        kH=kH, kW=kW,
        BLOCK_C_IN=32
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.kernel_size
        )