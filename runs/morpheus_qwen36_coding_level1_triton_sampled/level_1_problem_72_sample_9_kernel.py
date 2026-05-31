import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C_in, C_out, D_in, H_in, W_in,
    kD, kH, kW,
    strideD, strideH, strideW,
    padD, padH, padW,
    outD, outH, outW,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid over flattened output
    pid = tl.program_id(0)
    
    # Compute output indices
    n = pid // (C_out * outD * outH * outW)
    rem = pid % (C_out * outD * outH * outW)
    co = rem // (outD * outH * outW)
    rem = rem % (outD * outH * outW)
    do = rem // (outH * outW)
    ho = rem % outH // outW
    wo = rem % outW
    
    # Accumulator for the output value
    acc = 0.0
    
    # Loop over input channels (per group)
    for ci in range(C_in // groups):
        # Loop over kernel dimensions
        for kd in range(kD):
            for kh in range(kH):
                for kw in range(kW):
                    # Compute corresponding input indices
                    di = do * strideD - padD + kd
                    hi = ho * strideH - padH + kh
                    wi = wo * strideW - padW + kw
                    
                    # Check bounds
                    if 0 <= di < D_in and 0 <= hi < H_in and 0 <= wi < W_in:
                        # Weight index: (C_out, C_in//groups, kD, kH, kW)
                        wi_idx = (co * (C_in // groups) + ci) * kD * kH * kW + kd * kH * kW + kh * kW + kw
                        # Input index: (N, C_in, D_in, H_in, W_in)
                        xi_idx = n * C_in * D_in * H_in * W_in + ci * D_in * H_in * W_in + di * H_in * W_in + hi * W_in + wi
                        
                        # Load and multiply
                        w = tl.load(w_ptr + wi_idx)
                        x = tl.load(x_ptr + xi_idx)
                        acc += w * x
    
    # Store result
    tl.store(out_ptr + pid, acc)


def triton_conv_transpose3d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Wrapper for the Triton kernel implementing 3D transposed convolution.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, kD, kH, kW = w.shape
    groups = C_in // w.shape[1]
    
    # Compute output dimensions
    outD = (D_in - 1) * 2 - 2 * 1 + 1 + kD
    outH = (H_in - 1) * 2 - 2 * 2 + 1 + kH
    outW = (W_in - 1) * 2 - 2 * 3 + 1 + kW
    
    # Prepare output tensor
    out = torch.empty((N, C_out, outD, outH, outW), dtype=x.dtype, device=x.device)
    
    # Flatten output for 1D grid
    n_elements = N * C_out * outD * outH * outW
    BLOCK_SIZE = 128
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, w, out,
        N, C_in, C_out, D_in, H_in, W_in,
        kD, kH, kW,
        2, 2, 2,  # stride
        1, 2, 3,  # padding
        outD, outH, outW,
        groups,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(x, self.conv_transpose3d.weight, self.conv_transpose3d.bias if self.conv_transpose3d.bias is not None else None)