import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K, S, P,
    multiplier,
    has_bias,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_c, stride_w_kh, stride_w_kw,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # Decompose pid_0 into batch, output channel, and output height
    # pid_0 = b * (C_out * H_out) + c_out * H_out + h_out
    b = pid_0 // (C_out * H_out)
    rem = pid_0 % (C_out * H_out)
    c_out = rem // H_out
    h_out = rem % H_out

    # Map output channel to input channel (for depthwise groups=C_in)
    c_in = c_out // multiplier

    # Width block offsets
    w_out_offsets = pid_1 * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = w_out_offsets < W_out

    # Accumulator for the convolution
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Convolution loop over kernel height and width
    for kh in range(K):
        h_in = h_out * S + kh - P
        if h_in >= 0 and h_in < H_in:
            for kw in range(K):
                # Load weight for this specific output channel and kernel position
                # Weight shape: (C_out, 1, K, K)
                w_val = tl.load(w_ptr + c_out * stride_w_c + kh * stride_w_kh + kw * stride_w_kw)

                # Calculate input width offsets and mask
                w_in_offsets = w_out_offsets * S + kw - P
                mask_x = mask_w & (w_in_offsets >= 0) & (w_in_offsets < W_in)
                
                # Load input block
                # Input shape: (B, C_in, H_in, W_in)
                x_ptr_now = x_ptr + b * stride_x_b + c_in * stride_x_c + h_in * stride_x_h + w_in_offsets * stride_x_w
                x_val = tl.load(x_ptr_now, mask=mask_x, other=0.0)

                acc += x_val * w_val

    # Add bias if applicable
    if has_bias:
        bias_val = tl.load(b_ptr + c_out)
        acc += bias_val

    # Store output result
    # Output shape: (B, C_out, H_out, W_out)
    out_ptr_now = out_ptr + b * stride_out_b + c_out * stride_out_c + h_out * stride_out_h + w_out_offsets * stride_out_w
    tl.store(out_ptr_now, acc, mask=mask_w)


def triton_depthwise_conv2d(x, weight, bias, stride, padding):
    # Ensure inputs are contiguous on CUDA
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, C_in, H_in, W_in = x.shape
    C_out, _, K, _ = weight.shape
    S = stride
    P = padding
    
    H_out = (H_in + 2 * P - K) // S + 1
    W_out = (W_in + 2 * P - K) // S + 1
    multiplier = C_out // C_in

    out = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Strides for indexing
    stride_x_b = C_in * H_in * W_in
    stride_x_c = H_in * W_in
    stride_x_h = W_in
    stride_x_w = 1

    stride_w_c = 1 * K * K
    stride_w_kh = K
    stride_w_kw = 1

    stride_out_b = C_out * H_out * W_out
    stride_out_c = H_out * W_out
    stride_out_h = W_out
    stride_out_w = 1

    BLOCK_W = 32
    grid = (B * C_out * H_out, triton.cdiv(W_out, BLOCK_W))

    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, S, P,
        multiplier,
        bias is not None,
        stride_x_b, stride_x_c, stride_x_h, stride_x_w,
        stride_w_c, stride_w_kh, stride_w_kw,
        stride_out_b, stride_out_c, stride_out_h, stride_out_w,
        BLOCK_W=BLOCK_W,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        # Initialize weights to match nn.Conv2d(groups=in_channels)
        # Weight shape: (out_channels, 1, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The original architecture performs a depthwise convolution
        # We replace nn.Conv2d with our Triton wrapper
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding
        )