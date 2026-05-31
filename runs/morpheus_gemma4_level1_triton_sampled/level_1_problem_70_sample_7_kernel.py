import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, C_in, D, H, W,
    C_out, kD, kH, kW,
    S, P, Dil,
    D_out, H_out, W_out,
    G,
    BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_od = tl.program_id(2)
    pid_oh = tl.program_id(3)
    pid_ow_block = tl.program_id(4)

    # Output coordinates
    ow_offsets = pid_ow_block * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_ow = ow_offsets < W_out

    # Grouping logic
    C_out_per_group = C_out // G
    C_in_per_group = C_in // G
    oc_group = pid_oc // C_out_per_group
    oc_in_group = pid_oc % C_out_per_group
    ic_start = oc_group * C_in_per_group
    ic_end = ic_start + C_in_per_group

    # Accumulator for the output block
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels in the current group
    for ic in range(ic_start, ic_end):
        # Loop over kernel dimensions
        for kd in range(kD):
            id_val = pid_od + P - kd * Dil
            if id_val >= 0 and id_val % S == 0:
                id = id_val // S
                if id < D:
                    for kh in range(kH):
                        ih_val = pid_oh + P - kh * Dil
                        if ih_val >= 0 and ih_val % S == 0:
                            ih = ih_val // S
                            if ih < H:
                                for kw in range(kW):
                                    iw_val = ow_offsets + P - kw * Dil
                                    # Check if (ow + P - kw * Dil) is divisible by S
                                    # Since ow_offsets is a tensor, we use a mask
                                    # Note: In Triton, modulo on tensors is supported
                                    mask_iw = (iw_val >= 0) & (iw_val % S == 0)
                                    iw = iw_val // S
                                    mask_iw = mask_iw & (iw < W)
                                    
                                    # Load weight: w shape (C_in, C_out_per_group, kD, kH, kW)
                                    # Weight index: ic, oc_in_group, kd, kh, kw
                                    w_idx = ic * (C_out_per_group * kD * kH * kW) + \
                                            oc_in_group * (kD * kH * kW) + \
                                            kd * (kH * kW) + kh * kW + kw
                                    weight = tl.load(w_ptr + w_idx)

                                    # Load input: x shape (N, C_in, D, H, W)
                                    # Input index: pid_b, ic, id, ih, iw
                                    x_idx = pid_b * (C_in * D * H * W) + \
                                            ic * (D * H * W) + \
                                            id * (H * W) + \
                                            ih * W + \
                                            iw
                                    
                                    # Use mask for the width dimension
                                    val = tl.load(x_ptr + x_idx, mask=mask_ow & mask_iw, other=0.0)
                                    acc += val * weight

    # Add bias
    bias = tl.load(bias_ptr + pid_oc) if bias_ptr is not None else 0.0
    acc += bias

    # Store output: out shape (N, C_out, D_out, H_out, W_out)
    out_idx = pid_b * (C_out * D_out * H_out * W_out) + \
              pid_oc * (D_out * H_out * W_out) + \
              pid_od * (H_out * W_out) + \
              pid_oh * W_out + \
              ow_offsets
    
    tl.store(out_ptr + out_idx, acc, mask=mask_ow)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    # Input shapes
    N, C_in, D, H, W = x.shape
    C_in_w, C_out_per_group, kD, kH, kW = weight.shape
    C_out = C_out_per_group * groups

    # Assume scalar parameters for simplicity in kernel launch, as per common usage
    S = stride if isinstance(stride, int) else stride[0]
    P = padding if isinstance(padding, int) else padding[0]
    Dil = dilation if isinstance(dilation, int) else dilation[0]
    OP = output_padding if isinstance(output_padding, int) else output_padding[0]

    # Calculate output dimensions
    D_out = (D - 1) * S - 2 * P + Dil * (kD - 1) + 1 + OP
    H_out = (H - 1) * S - 2 * P + Dil * (kH - 1) + 1 + OP
    W_out = (W - 1) * S - 2 * P + Dil * (kW - 1) + 1 + OP

    out = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    BLOCK_W = 32
    grid = (N, C_out, D_out, H_out, (W_out + BLOCK_W - 1) // BLOCK_W)

    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, C_in, D, H, W,
        C_out, kD, kH, kW,
        S, P, Dil,
        D_out, H_out, W_out,
        groups,
        BLOCK_W=BLOCK_W,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Use the original layer to manage weights and bias
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
            stride=stride, padding=padding, output_padding=output_padding, 
            dilation=dilation, groups=groups, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the internal ConvTranspose3d layer
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias if self.conv_transpose3d.bias is not None else None
        
        return triton_conv_transpose3d(
            x, weight, bias, 
            self.stride, self.padding, self.output_padding, 
            self.dilation, self.groups
        )