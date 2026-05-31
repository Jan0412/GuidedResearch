import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W,
    OC, KH, KW,
    OH, OW,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    groups,
    BLOCK_SIZE_OW: tl.constexpr,
):
    # Parallelize over N, OC, OH, and tiled OW
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow_block = tl.program_id(3)

    # Output width offsets
    ow_offsets = pid_ow_block * BLOCK_SIZE_OW + tl.arange(0, BLOCK_SIZE_OW)
    mask_ow = ow_offsets < OW

    # Group logic
    oc_per_group = OC // groups
    group_id = pid_oc // oc_per_group
    c_start = group_id * (C // groups)
    c_per_group = C // groups

    # Accumulator for the block of OW
    acc = tl.zeros((BLOCK_SIZE_OW,), dtype=tl.float32)

    # Loop over input channels (within group)
    for c_rel in range(c_per_group):
        c = c_start + c_rel
        # Loop over kernel height
        for kh in range(KH):
            # Input height index
            ih = pid_oh * stride_h + kh * dil_h - pad_h
            mask_ih = (ih >= 0) & (ih < H)
            
            # Loop over kernel width
            for kw in range(KW):
                # Input width indices
                iw = ow_offsets * stride_w + kw * dil_w - pad_w
                mask_iw = (iw >= 0) & (iw < W)
                mask = mask_ow & mask_ih & mask_iw

                # Load weight (scalar for this block)
                w_val = tl.load(w_ptr + pid_oc * c_per_group * KH * KW + c_rel * KH * KW + kh * KW + kw)
                
                # Load input values (vector over OW)
                # x_ptr: (N, C, H, W)
                x_val = tl.load(x_ptr + pid_n * C * H * W + c * H * W + ih * W + iw, mask=mask, other=0.0)
                
                acc += x_val * w_val

    # Add bias
    bias_val = tl.load(b_ptr + pid_oc)
    acc += bias_val

    # Store output
    # out_ptr: (N, OC, OH, OW)
    out_offset = pid_n * OC * OH * OW + pid_oc * OH * OW + pid_oh * OW + ow_offsets
    tl.store(out_ptr + out_offset, acc, mask=mask_ow)


def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    # Ensure inputs are contiguous and on CUDA
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    else:
        # Create a dummy zero bias if none provided to simplify kernel
        bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)

    N, C, H, W = x.shape
    OC, C_per_group, KH, KW = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    # Calculate output dimensions
    OH = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    OW = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_OW = 32
    grid = (N, OC, OH, (OW + BLOCK_SIZE_OW - 1) // BLOCK_SIZE_OW)

    conv2d_kernel[grid](
        x, weight, bias, out,
        N, C, H, W,
        OC, KH, KW,
        OH, OW,
        sh, sw,
        ph, pw,
        dh, dw,
        groups,
        BLOCK_SIZE_OW=BLOCK_SIZE_OW,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the nn.Conv2d to manage the learnable parameters (weight and bias)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use our custom Triton implementation instead of the standard PyTorch forward pass
        return triton_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )