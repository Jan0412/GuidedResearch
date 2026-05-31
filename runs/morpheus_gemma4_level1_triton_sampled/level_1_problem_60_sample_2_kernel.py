import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, W, H, D,
    OC, IC_g, KW, KH, KD,
    WO, HO, DO,
    stride_w, stride_h, stride_d,
    pad_w, pad_h, pad_d,
    dil_w, dil_h, dil_d,
    groups,
    X_S_B, X_S_IC, X_S_W, X_S_H, X_S_D,
    W_S_OC, W_S_ICg, W_S_KW, W_S_KH, W_S_KD,
    OUT_S_B, OUT_S_OC, OUT_S_WO, OUT_S_HO, OUT_S_DO,
):
    # Map program ID to output coordinates
    # grid = (B * OC, WO, HO, DO)
    pid_b_oc = tl.program_id(0)
    pid_wo = tl.program_id(1)
    pid_ho = tl.program_id(2)
    pid_do = tl.program_id(3)

    b = pid_b_oc // OC
    oc = pid_b_oc % OC

    # Calculate the base pointer for the output element
    out_offset = (b * OUT_S_B + 
                  oc * OUT_S_OC + 
                  pid_wo * OUT_S_WO + 
                  pid_ho * OUT_S_HO + 
                  pid_do * OUT_S_DO)
    
    acc = 0.0
    
    # Grouping logic: determine which input channels this output channel looks at
    ic_start = (oc // groups) * IC_g

    # Direct convolution loops
    # Note: In a production kernel, we would tile these loops for performance.
    # For correctness and functionality in FP32, we iterate through the kernel and input channels.
    for ic_g in range(0, IC_g):
        ic = ic_start + ic_g
        for kw in range(0, KW):
            iw = pid_wo * stride_w + kw * dil_w - pad_w
            if iw < 0 or iw >= W:
                continue
            for kh in range(0, KH):
                ih = pid_ho * stride_h + kh * dil_h - pad_h
                if ih < 0 or ih >= H:
                    continue
                for kd in range(0, KD):
                    id_ = pid_do * stride_d + kd * dil_d - pad_d
                    if id_ < 0 or id_ >= D:
                        continue
                    
                    # Load input value
                    x_off = (b * X_S_B + ic * X_S_IC + iw * X_S_W + ih * X_S_H + id_ * X_S_D)
                    x_val = tl.load(x_ptr + x_off)
                    
                    # Load weight value
                    w_off = (oc * W_S_OC + ic_g * W_S_ICg + kw * W_S_KW + kh * W_S_KH + kd * W_S_KD)
                    w_val = tl.load(w_ptr + w_off)
                    
                    acc += x_val * w_val

    # Add bias if applicable
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc)
        acc += bias_val

    tl.store(out_ptr + out_offset, acc)

def triton_conv3d(x, weight, bias, stride, padding, dilation, groups):
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, IC, W, H, D = x.shape
    OC, IC_g, KW, KH, KD = weight.shape
    
    # Handle stride, padding, dilation as tuples
    def to_tuple(val):
        if isinstance(val, int):
            return (val, val, val)
        return val

    sw, sh, sd = to_tuple(stride)
    pw, ph, pd = to_tuple(padding)
    dw, dh, dd = to_tuple(dilation)

    # Calculate output dimensions
    WO = (W + 2 * pw - (KW - 1) * dw - 1) // sw + 1
    HO = (H + 2 * ph - (KH - 1) * dh - 1) // sh + 1
    DO = (D + 2 * pd - (KD - 1) * dd - 1) // sd + 1

    out = torch.empty((B, OC, WO, HO, DO), device=x.device, dtype=x.dtype)
    
    # Strides for 5D tensors
    X_S_B = IC * W * H * D
    X_S_IC = W * H * D
    X_S_W = H * D
    X_S_H = D
    X_S_D = 1

    W_S_OC = IC_g * KW * KH * KD
    W_S_ICg = KW * KH * KD
    W_S_KW = KH * KD
    W_S_KH = KD
    W_S_KD = 1

    OUT_S_B = OC * WO * HO * DO
    OUT_S_OC = WO * HO * DO
    OUT_S_WO = HO * DO
    OUT_S_HO = DO
    OUT_S_DO = 1

    grid = (B * OC, WO, HO, DO)

    conv3d_kernel[grid](
        x, weight, bias, out,
        B, IC, W, H, D,
        OC, IC_g, KW, KH, KD,
        WO, HO, DO,
        sw, sh, sd,
        pw, ph, pd,
        dw, dh, dd,
        groups,
        X_S_B, X_S_IC, X_S_W, X_S_H, X_S_D,
        W_S_OC, W_S_ICg, W_S_KW, W_S_KH, W_S_KD,
        OUT_S_B, OUT_S_OC, OUT_S_WO, OUT_S_HO, OUT_S_DO,
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size # (kw, kh, kd)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias as parameters to mimic nn.Conv3d
        # Weight shape: (out_channels, in_channels // groups, kw, kh, kd)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom Triton implementation
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )