import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_k1_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W,
    H_out, W_out,
    K, stride, padding, dilation,
    S_XN, S_XC, S_XH, S_XW,
    S_WN, S_WC, S_WK, S_WW,
    S_ON, S_OC, S_OH, S_OW,
    BLOCK_H: tl.constexpr,
):
    # Each program handles one (batch, channel, width_out) and a block of height_out
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Decode pid to n, c, w_out
    # pid = n * (C * W_out) + c * W_out + w_out
    n = pid // (C * W_out)
    rem = pid % (C * W_out)
    c = rem // W_out
    w_out = rem % W_out

    # height offsets for this block
    h_out_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_out_offsets < H_out

    # Calculate the constant input width index for this program
    w_in = w_out * stride + padding

    # Accumulator for the output block
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over the kernel height
    for k in range(K):
        # Calculate input height indices for the current kernel element
        h_in_offsets = h_out_offsets * stride + padding + k * dilation
        
        # Mask for valid input coordinates
        mask = mask_h & (h_in_offsets >= 0) & (h_in_offsets < H) & (w_in >= 0) & (w_in < W)
        
        # Load input values: x[n, c, h_in, w_in]
        # Pointer arithmetic: x_ptr + n*S_XN + c*S_XC + h_in*S_XH + w_in*S_XW
        x_offset = n * S_XN + c * S_XC + h_in_offsets * S_XH + w_in * S_XW
        x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        
        # Load weight: w[c, 0, k, 0]
        w_offset = c * S_WC + k * S_WK
        w_val = tl.load(w_ptr + w_offset)
        
        acc += x_val * w_val

    # Add bias if available
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + c)
        acc += bias_val

    # Store output: out[n, c, h_out, w_out]
    out_offset = n * S_ON + c * S_OC + h_out_offsets * S_OH + w_out * S_OW
    tl.store(out_ptr + out_offset, acc, mask=mask_h)


def triton_depthwise_conv2d_k1(x, weight, bias=None, stride=1, padding=0, dilation=1):
    # x: (N, C, H, W)
    # weight: (C, 1, K, 1)
    # bias: (C,)
    N, C, H, W = x.shape
    K = weight.shape[2]
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - 1) // stride + 1
    
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Strides
    S_XN, S_XC, S_XH, S_XW = x.stride()
    S_WN, S_WC, S_WK, S_WW = weight.stride()
    S_ON, S_OC, S_OH, S_OW = out.stride()
    
    BLOCK_H = 32
    grid = (N * C * W_out, triton.cdiv(H_out, BLOCK_H))
    
    depthwise_conv2d_k1_kernel[grid](
        x, weight, bias, out,
        N, C, H, W,
        H_out, W_out,
        K, stride, padding, dilation,
        S_XN, S_XC, S_XH, S_XW,
        S_WN, S_WC, S_WK, S_WW,
        S_ON, S_OC, S_OH, S_OW,
        BLOCK_H=BLOCK_H
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution with a square input and an asymmetric kernel (K, 1).
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Use nn.Conv2d to manage parameters easily, but we'll bypass its forward
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=(kernel_size, 1), 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=in_channels, 
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are contiguous for the Triton kernel
        x = x.contiguous()
        weight = self.conv2d.weight.contiguous()
        bias = self.conv2d.bias.contiguous() if self.conv2d.bias is not None else None
        
        return triton_depthwise_conv2d_k1(
            x, 
            weight, 
            bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )