import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, Cin, Cout, Din, Hin, Win,
    Dout, Hout, Wout,
    K, stride, padding, dilation,
    sN, sCin, sDin, sHin, sWin,
    swCin, swCout, swKd, swKh, swKw,
    soN, soCout, soDout, soHout, soWout,
    BLOCK_COUT: tl.constexpr,
):
    # Program IDs
    n = tl.program_id(0)
    d = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    cout_block_id = tl.program_id(4)

    # Output channel range
    cout_offsets = cout_block_id * BLOCK_COUT + tl.arange(0, BLOCK_COUT)
    mask_cout = cout_offsets < Cout

    # Accumulator for the block of output channels
    acc = tl.zeros([BLOCK_COUT], dtype=tl.float32)

    # Loop over input channels and kernel dimensions
    # In Triton, we use while loops for non-constexpr loop bounds
    cin = 0
    while cin < Cin:
        kd = 0
        while kd < K:
            # Calculate input depth index
            # d_out = d_in * stride - padding + kd * dilation
            # d_in = (d_out + padding - kd * dilation) / stride
            din_num = d + padding - kd * dilation
            if din_num >= 0 and din_num % stride == 0:
                din = din_num // stride
                if din < Din:
                    kh = 0
                    while kh < K:
                        hin_num = h + padding - kh * dilation
                        if hin_num >= 0 and hin_num % stride == 0:
                            hin = hin_num // stride
                            if hin < Hin:
                                kw = 0
                                while kw < K:
                                    win_num = w + padding - kw * dilation
                                    if win_num >= 0 and win_num % stride == 0:
                                        win = win_num // stride
                                        if win < Win:
                                            # Load input value: x shape (N, Cin, Din, Hin, Win)
                                            x_val = tl.load(x_ptr + n * sN + cin * sCin + din * sDin + hin * sHin + win * sWin)
                                            
                                            # Load weight values for the block of Cout: w shape (Cin, Cout, K, K, K)
                                            w_ptr_base = w_ptr + cin * swCin + cout_offsets * swCout + kd * swKd + kh * swKh + kw * swKw
                                            w_vals = tl.load(w_ptr_base, mask=mask_cout)
                                            acc += x_val * w_vals
                                    kw += 1
                        kh += 1
            kd += 1
        cin += 1

    # Add bias if provided
    if b_ptr is not None:
        bias_vals = tl.load(b_ptr + cout_offsets, mask=mask_cout)
        acc += bias_vals

    # Store results: out shape (N, Cout, Dout, Hout, Wout)
    out_ptr_base = out_ptr + n * soN + cout_offsets * soCout + d * soDout + h * soHout + w * soWout
    tl.store(out_ptr_base, acc, mask=mask_cout)


def triton_conv_transpose3d(x, weight, bias, stride, padding, dilation):
    # Input shapes
    N, Cin, Din, Hin, Win = x.shape
    Cin_w, Cout, Kd, Kh, Kw = weight.shape
    K = Kd # Assume square kernel as per model description

    # Calculate output dimensions
    Dout = (Din - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    Hout = (Hin - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    Wout = (Win - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    out = torch.empty((N, Cout, Dout, Hout, Wout), device=x.device, dtype=x.dtype)
    
    # Strides
    sN, sCin, sDin, sHin, sWin = x.stride()
    swCin, swCout, swKd, swKh, swKw = weight.stride()
    soN, soCout, soDout, soHout, soWout = out.stride()
    
    BLOCK_COUT = 32
    grid = (N, Dout, Hout, Wout, (Cout + BLOCK_COUT - 1) // BLOCK_COUT)
    
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, Cin, Cout, Din, Hin, Win,
        Dout, Hout, Wout,
        K, stride, padding, dilation,
        sN, sCin, sDin, sHin, sWin,
        swCin, swCout, swKd, swKh, swKw,
        soN, soCout, soDout, soHout, soWout,
        BLOCK_COUT=BLOCK_COUT,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Maintain parameters using standard PyTorch layers for initialization and storage
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=(kernel_size, kernel_size, kernel_size), 
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous to simplify stride calculations in Triton
        x = x.contiguous()
        weight = self.conv_transpose3d.weight.contiguous()
        bias = self.conv_transpose3d.bias.contiguous() if self.conv_transpose3d.bias is not None else None
        
        return triton_conv_transpose3d(
            x, 
            weight, 
            bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )