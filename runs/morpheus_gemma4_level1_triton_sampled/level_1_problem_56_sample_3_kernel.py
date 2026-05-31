import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, Cin, H, W,
    Cout, Cin_group, kH, kW,
    Hout, Wout,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    groups,
    BLOCK_C_OUT: tl.constexpr,
):
    # Program ID for output spatial location and batch
    pid_out = tl.program_id(0)
    # Program ID for output channel block
    pid_cout = tl.program_id(1)

    # Decode output coordinates
    b = pid_out // (Hout * Wout)
    rem = pid_out % (Hout * Wout)
    h_out = rem // Wout
    w_out = rem % Wout

    # Output channel range for this block
    cout_range = pid_cout * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)
    mask_cout = cout_range < Cout

    # Accumulator for the convolution result
    acc = tl.zeros([BLOCK_C_OUT], dtype=tl.float32)

    Cout_group = Cout // groups

    # Iterate over the input channel group, kernel height, and kernel width
    for cin_idx in range(Cin_group):
        for kh in range(kH):
            for kw in range(kW):
                # Calculate input coordinates with stride, padding, and dilation
                h_in = h_out * stride_h + kh * dilation_h - padding_h
                w_in = w_out * stride_w + kw * dilation_w - padding_w
                
                # Boundary check for input padding
                mask_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                
                # Determine which input channel to use based on the output channel's group
                gid = cout_range // Cout_group
                x_chan = gid * Cin_group + cin_idx
                
                # Calculate pointers
                # x shape: (B, Cin, H, W)
                x_off = b * (Cin * H * W) + x_chan * (H * W) + h_in * W + w_in
                # w shape: (Cout, Cin_group, kH, kW)
                w_off = cout_range * (Cin_group * kH * kW) + cin_idx * (kH * kW) + kh * kW + kw
                
                # Load values
                val_x = tl.load(x_ptr + x_off, mask=mask_in & mask_cout, other=0.0)
                val_w = tl.load(w_ptr + w_off, mask=mask_cout, other=0.0)
                
                acc += val_x * val_w

    # Add bias if provided
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + cout_range, mask=mask_cout, other=0.0)
        acc += bias_val

    # Store the result
    # out shape: (B, Cout, Hout, Wout)
    out_off = b * (Cout * Hout * Wout) + cout_range * (Hout * Wout) + h_out * Wout + w_out
    tl.store(out_ptr + out_off, acc, mask=mask_cout)


def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    # Ensure tensors are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, Cin, H, W = x.shape
    Cout, Cin_group, kH, kW = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    # Calculate output dimensions
    Hout = (H + 2 * ph - dh * (kH - 1) - 1) // sh + 1
    Wout = (W + 2 * pw - dw * (kW - 1) - 1) // sw + 1

    out = torch.empty((B, Cout, Hout, Wout), device=x.device, dtype=x.dtype)
    
    BLOCK_C_OUT = 32
    grid = (B * Hout * Wout, triton.cdiv(Cout, BLOCK_C_OUT))

    conv2d_kernel[grid](
        x, weight, bias, out,
        B, Cin, H, W,
        Cout, Cin_group, kH, kW,
        Hout, Wout,
        sh, sw,
        ph, pw,
        dh, dw,
        groups,
        BLOCK_C_OUT=BLOCK_C_OUT,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Use nn.Conv2d to initialize weights and bias correctly
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call our custom Triton-based convolution
        return triton_conv2d(
            x, 
            self.conv.weight, 
            self.conv.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )