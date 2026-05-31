import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W, KH,
    stride, padding, dilation,
    H_out, W_out,
    BLOCK_H: tl.constexpr,
):
    # pid 0: index into (batch, channel, width_out)
    # pid 1: index into height_out blocks
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # Decode pid_0 to n, c, w_out
    n = pid_0 // (C * W_out)
    rem = pid_0 % (C * W_out)
    c = rem // W_out
    w_out = rem % W_out

    # Height block offsets
    h_out_start = pid_1 * BLOCK_H
    h_out_offsets = h_out_start + tl.arange(0, BLOCK_H)
    mask_h = h_out_offsets < H_out

    # Load bias for the current channel
    bias = tl.load(b_ptr + c) if b_ptr is not None else 0.0

    # Weight pointer for this channel: shape (C, 1, KH, 1)
    w_channel_ptr = w_ptr + c * KH

    # Initialize accumulator with bias
    acc = tl.full((BLOCK_H,), bias, dtype=tl.float32)

    # Convolution loop over the kernel height (KH)
    for kh in range(KH):
        # Calculate input height indices for the block of h_out
        h_in_offsets = h_out_offsets * stride + kh * dilation - padding
        # Calculate input width index
        w_in = w_out * stride - padding

        # Mask for input height and width boundaries
        mask_in = mask_h & (h_in_offsets >= 0) & (h_in_offsets < H) & (w_in >= 0) & (w_in < W)

        # Compute pointer to input elements
        # x shape: (N, C, H, W)
        x_offset = n * (C * H * W) + c * (H * W) + h_in_offsets * W + w_in
        x_vals = tl.load(x_ptr + x_offset, mask=mask_in, other=0.0)

        # Load the weight for the current kh
        weight = tl.load(w_channel_ptr + kh)

        acc += x_vals * weight

    # Store result in output tensor
    # out shape: (N, C, H_out, W_out)
    out_offset = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out_offsets * W_out + w_out
    tl.store(out_ptr + out_offset, acc, mask=mask_h)


def triton_depthwise_conv2d(x, weight, bias, stride, padding, dilation):
    # x: (N, C, H, W)
    # weight: (C, 1, KH, 1)
    # bias: (C,) or None
    N, C, H, W = x.shape
    KH = weight.shape[2]
    
    # Calculate output dimensions
    # For kernel width = 1, W_out formula simplifies
    H_out = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - 1) // stride + 1
    
    x = x.contiguous().cuda()
    weight = weight.contiguous().cuda()
    if bias is not None:
        bias = bias.contiguous().cuda()
    
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    
    BLOCK_H = 32
    grid = (N * C * W_out, triton.cdiv(H_out, BLOCK_H))
    
    depthwise_conv_kernel[grid](
        x, weight, bias, out,
        N, C, H, W, KH,
        stride, padding, dilation,
        H_out, W_out,
        BLOCK_H=BLOCK_H
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution with an asymmetric kernel using Triton.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # We use nn.Conv2d to manage parameters, but we'll use our Triton kernel in forward()
        self.conv2d = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=(kernel_size, 1), 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=in_channels, 
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the Triton optimized kernel instead of the PyTorch conv2d operator
        return triton_depthwise_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )