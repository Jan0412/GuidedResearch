import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C, H, W,
    H_out, W_out,
    s_h, p_h, d_h,
    s_w, p_w, d_w,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    K: tl.constexpr,
    BLOCK_H_OUT: tl.constexpr,
):
    # program_id(0) handles the (batch, channel, width_out) dimension
    pid_bcw = tl.program_id(0)
    # program_id(1) handles a block of the output height dimension
    pid_h = tl.program_id(1)

    # Decode pid_bcw to get batch, channel, and width_out indices
    w_out = pid_bcw % W_out
    tmp = pid_bcw // W_out
    c = tmp % C
    b = tmp // C

    # Calculate the corresponding input width index (kernel width is 1)
    w_in = s_w * w_out - p_w
    w_mask = (w_in >= 0) & (w_in < W)

    # Calculate the range of output height indices for this block
    h_out_offsets = pid_h * BLOCK_H_OUT + tl.arange(0, BLOCK_H_OUT)
    mask_h = h_out_offsets < H_out

    # Accumulator for the convolution result
    acc = tl.zeros([BLOCK_H_OUT], dtype=tl.float32)

    # Base pointer for weights of the current channel
    # Weights are stored as (C, 1, K, 1), flattened to (C * K)
    w_base = w_ptr + c * K

    # Perform the 1D convolution along the height dimension
    for i in range(K):
        # Calculate input height indices for the current kernel element
        h_in = s_h * h_out_offsets + i * d_h - p_h
        mask_h_in = mask_h & (h_in >= 0) & (h_in < H) & w_mask
        
        # Calculate pointer offsets for the input tensor x (B, C, H, W)
        x_offsets = b * stride_x_b + c * stride_x_c + h_in * stride_x_h + w_in * stride_x_w
        val = tl.load(x_ptr + x_offsets, mask=mask_h_in, other=0.0)
        
        # Load weight for current channel and kernel index
        weight = tl.load(w_base + i)
        acc += val * weight

    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + c)
        acc += bias

    # Calculate pointer offsets for the output tensor (B, C, H_out, W_out)
    out_offsets = b * stride_out_b + c * stride_out_c + h_out_offsets * stride_out_h + w_out * stride_out_w
    tl.store(out_ptr + out_offsets, acc, mask=mask_h)


def triton_depthwise_conv(x, weight, bias, stride, padding, dilation):
    # Input shapes
    B, C, H, W = x.shape
    # Weight shape (C, 1, K, 1)
    K = weight.shape[2]
    
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation

    # Calculate output dimensions
    H_out = (H + 2 * p_h - d_h * (K - 1) - 1) // s_h + 1
    W_out = (W + 2 * p_w - d_w * (1 - 1) - 1) // s_w + 1

    # Ensure tensors are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous().view(C, K)
    if bias is not None:
        bias = bias.contiguous()

    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Strides for index calculation
    stride_x_b = C * H * W
    stride_x_c = H * W
    stride_x_h = W
    stride_x_w = 1

    stride_out_b = C * H_out * W_out
    stride_out_c = H_out * W_out
    stride_out_h = W_out
    stride_out_w = 1

    # Kernel configuration
    BLOCK_H_OUT = 32
    grid = (B * C * W_out, (H_out + BLOCK_H_OUT - 1) // BLOCK_H_OUT)

    depthwise_conv_kernel[grid](
        x, weight, bias, out,
        B, C, H, W,
        H_out, W_out,
        s_h, p_h, d_h,
        s_w, p_w, d_w,
        stride_x_b, stride_x_c, stride_x_h, stride_x_w,
        stride_out_b, stride_out_c, stride_out_h, stride_out_w,
        K=K,
        BLOCK_H_OUT=BLOCK_H_OUT,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep the original Conv2d to manage parameters and initialization
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
        # Extract parameters from the Conv2d layer
        weight = self.conv2d.weight
        bias = self.conv2d.bias if self.conv2d.bias is not None else None
        stride = self.conv2d.stride
        padding = self.conv2d.padding
        dilation = self.conv2d.dilation

        # Call the Triton-optimized implementation
        return triton_depthwise_conv(x, weight, bias, stride, padding, dilation)