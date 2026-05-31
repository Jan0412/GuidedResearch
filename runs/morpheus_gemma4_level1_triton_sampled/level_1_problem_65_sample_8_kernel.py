import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, OC, HI, WI, KH, KW, HO, WO,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_b_oc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Calculate batch and output channel
    b = pid_b_oc // OC
    oc = pid_b_oc % OC

    # Calculate output height and width offsets
    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W
    oh = oh_start + tl.arange(0, BLOCK_H)
    ow = ow_start + tl.arange(0, BLOCK_W)

    # Output boundary mask
    out_mask = (oh[:, None] < HO) & (ow[None, :] < WO)

    # Accumulator for the output tile
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Iterate over input channels and kernel dimensions
    # For ConvTranspose2d with stride=1, padding=0, the operation is
    # Y[b, oc, oh, ow] = sum_{ic, kh, kw} X[b, ic, oh-kh, ow-kw] * W[ic, oc, kh, kw]
    for ic in range(0, IC):
        for kh in range(0, KH):
            for kw in range(0, KW):
                # Input coordinates
                ih = oh - kh
                iw = ow - kw

                # Mask for input boundaries
                in_mask = (ih[:, None] >= 0) & (ih[:, None] < HI) & \
                          (iw[None, :] >= 0) & (iw[None, :] < WI)

                # Load input tile: x[b, ic, ih, iw]
                # Offset: b * (IC*HI*WI) + ic * (HI*WI) + ih * WI + iw
                x_off = b * (IC * HI * WI) + ic * (HI * WI) + \
                        ih[:, None] * WI + iw[None, :]
                x_val = tl.load(x_ptr + x_off, mask=in_mask, other=0.0)

                # Load weight: w[ic, oc, kh, kw]
                # Offset: ic * (OC*KH*KW) + oc * (KH*KW) + kh * KW + kw
                w_off = ic * (OC * KH * KW) + oc * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off)

                acc += x_val * w_val

    # Add bias if provided
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc)
        acc += bias_val

    # Store output tile: out[b, oc, oh, ow]
    # Offset: b * (OC*HO*WO) + oc * (HO*WO) + oh * WO + ow
    out_off = b * (OC * HO * WO) + oc * (HO * WO) + \
              oh[:, None] * WO + ow[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def triton_conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0):
    """
    Triton wrapper for ConvTranspose2d. 
    Optimized for stride=1, padding=0, groups=1.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous
    x = x.contiguous().float()
    weight = weight.contiguous().float()
    if bias is not None:
        bias = bias.contiguous().float()

    B, IC, HI, WI = x.shape
    IC_w, OC, KH, KW = weight.shape
    
    # Calculate output dimensions for stride=1, padding=0, output_padding=0
    # H_out = (H_in - 1) * stride - 2 * padding + (K - 1) + output_padding + 1
    HO = (HI - 1) * stride - 2 * padding + (KH - 1) + output_padding + 1
    WO = (WI - 1) * stride - 2 * padding + (KW - 1) + output_padding + 1

    out = torch.empty((B, OC, HO, WO), device=x.device, dtype=torch.float32)

    BLOCK_H = 16
    BLOCK_W = 16

    # Grid: (Batch * OutChannels, OutHeight / BLOCK_H, OutWidth / BLOCK_W)
    grid = (B * OC, (HO + BLOCK_H - 1) // BLOCK_H, (WO + BLOCK_W - 1) // BLOCK_W)

    conv_transpose_kernel[grid](
        x, weight, bias, out,
        B, IC, OC, HI, WI, KH, KW, HO, WO,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized transposed 2D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the original layer to manage parameters (weights and bias)
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding, 
            groups=groups, bias=bias
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the Triton implementation instead of the PyTorch forward pass
        # Note: This implementation currently supports groups=1, stride=1, padding=0
        return triton_conv_transpose2d(
            x, 
            self.conv_transpose2d.weight, 
            self.conv_transpose2d.bias, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding
        )