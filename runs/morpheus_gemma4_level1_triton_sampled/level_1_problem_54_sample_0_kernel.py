import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    B, Cin, D, H, W,
    Cout, Cin_g, kD, kH, kW,
    sD, sH, sW, pD, pH, pW,
    dilD, dilH, dilW,
    G, Dout, Hout, Wout,
    BLOCK_CIN: tl.constexpr,
):
    # Calculate output coordinates
    idx = tl.program_id(0)
    
    # Decompose index to (n, c_out, d_out, h_out, w_out)
    w_out = idx % Wout
    rem = idx // Wout
    h_out = rem % Hout
    rem //= Hout
    d_out = rem % Dout
    rem //= Dout
    c_out = rem % Cout
    n = rem // Cout

    # Group logic: find the range of input channels this output channel is connected to
    cout_per_group = Cout // G
    group_id = c_out // cout_per_group
    cin_start = group_id * Cin_g

    # Initialize accumulator
    acc = 0.0

    # Loop over input channels in blocks
    for c_in_g_start in range(0, Cin_g, BLOCK_CIN):
        c_in_g_offsets = c_in_g_start + tl.arange(0, BLOCK_CIN)
        c_in_mask = c_in_g_offsets < Cin_g
        
        # Global input channel index
        c_in = cin_start + c_in_g_offsets

        # Loop over the kernel spatial dimensions
        # Note: kD, kH, kW are usually small, so we can use Python loops for unrolling
        for kd in range(kD):
            for kh in range(kH):
                for kw in range(kW):
                    # Calculate input spatial coordinates
                    d_in = d_out * sD + kd * dilD - pD
                    h_in = h_out * sH + kh * dilH - pH
                    w_in = w_out * sW + kw * dilW - pW

                    # Bounds check for input spatial dimensions
                    if d_in >= 0 and d_in < D and h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                        # Load input: (B, Cin, D, H, W)
                        # index = n*Cin*D*H*W + c_in*D*H*W + d_in*H*W + h_in*W + w_in
                        x_off = n * (Cin * D * H * W) + c_in * (D * H * W) + d_in * (H * W) + h_in * W + w_in
                        x_val = tl.load(x_ptr + x_off, mask=c_in_mask, other=0.0)

                        # Load weight: (Cout, Cin_g, kD, kH, kW)
                        # index = c_out*Cin_g*kD*kH*kW + c_in_g*kD*kH*kW + kd*kH*kW + kh*kW + kw
                        w_off = c_out * (Cin_g * kD * kH * kW) + c_in_g_offsets * (kD * kH * kW) + \
                                kd * (kH * kW) + kh * kW + kw
                        w_val = tl.load(weight_ptr + w_off, mask=c_in_mask, other=0.0)

                        acc += x_val * w_val

    # Add bias if it exists
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + c_out)
        acc += bias_val

    # Store result: (B, Cout, Dout, Hout, Wout)
    out_off = n * (Cout * Dout * Hout * Wout) + c_out * (Dout * Hout * Wout) + \
              d_out * (Hout * Wout) + h_out * Wout + w_out
    tl.store(out_ptr + out_off, acc)

def triton_conv3d(x, weight, bias, stride, padding, dilation, groups):
    B, Cin, D, H, W = x.shape
    Cout, Cin_g, kD, kH, kW = weight.shape
    
    sD = sH = sW = stride
    pD = pH = pW = padding
    dilD = dilH = dilW = dilation
    
    Dout = (D + 2 * pD - dilD * (kD - 1) - 1) // sD + 1
    Hout = (H + 2 * pH - dilH * (kH - 1) - 1) // sH + 1
    Wout = (W + 2 * pW - dilW * (kW - 1) - 1) // sW + 1
    
    out = torch.empty((B, Cout, Dout, Hout, Wout), device=x.device, dtype=x.dtype)
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    out = out.contiguous()

    grid = (B * Cout * Dout * Hout * Wout,)
    
    conv3d_kernel[grid](
        x, weight, bias, out,
        B, Cin, D, H, W,
        Cout, Cin_g, kD, kH, kW,
        sD, sH, sW, pD, pH, pW,
        dilD, dilH, dilW,
        groups, Dout, Hout, Wout,
        BLOCK_CIN=32
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_enabled = bias

        # Use nn.Parameter to store weights and bias for easy management
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input is on GPU and FP32
        x = x.cuda().float()
        weight = self.weight.cuda().float()
        bias = self.bias.cuda().float() if self.bias is not None else None
        
        return triton_conv3d(
            x, weight, bias, 
            self.stride, self.padding, self.dilation, self.groups
        )