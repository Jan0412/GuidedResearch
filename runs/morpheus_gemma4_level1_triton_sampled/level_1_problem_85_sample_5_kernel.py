import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, C, H, W,
    KH, KW,
    H_out, W_out,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_c, stride_w_kh, stride_w_kw,
    stride_y_b, stride_y_c, stride_y_h, stride_y_w,
    has_bias,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Map pid_bc to batch and channel
    batch_idx = pid_bc // C
    chan_idx = pid_bc % C

    # Output offsets
    h_out_offsets = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_out_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Masks for output boundaries
    mask_h = h_out_offsets < H_out
    mask_w = w_out_offsets < W_out

    # Accumulator for the output block
    acc = tl.zeros([BLOCK_SIZE_H, BLOCK_SIZE_W], dtype=tl.float32)

    # Iterate over the kernel height and width
    for kh in range(KH):
        for kw in range(KW):
            # Calculate input coordinates
            h_in = h_out_offsets * stride_h + kh * dilation_h - padding_h
            w_in = w_out_offsets * stride_w + kw * dilation_w - padding_w

            # Load weight for this channel and kernel position
            # Weight shape: (C, 1, KH, KW)
            w_val = tl.load(w_ptr + chan_idx * stride_w_c + kh * stride_w_kh + kw * stride_w_kw)

            # Calculate input pointers for the block
            # x shape: (B, C, H, W)
            # h_in is (BLOCK_SIZE_H,), w_in is (BLOCK_SIZE_W,)
            # We use broadcasting to create a (BLOCK_SIZE_H, BLOCK_SIZE_W) pointer grid
            x_ptr_in = (
                x_ptr + 
                batch_idx * stride_x_b + 
                chan_idx * stride_x_c + 
                h_in[:, None] * stride_x_h + 
                w_in[None, :] * stride_x_w
            )

            # Mask for input boundaries
            mask_in = (
                mask_h[:, None] & 
                mask_w[None, :] & 
                (h_in[:, None] >= 0) & (h_in[:, None] < H) & 
                (w_in[None, :] >= 0) & (w_in[None, :] < W)
            )

            # Load input values and accumulate
            x_val = tl.load(x_ptr_in, mask=mask_in, other=0.0)
            acc += x_val * w_val

    # Add bias if applicable
    if has_bias:
        bias_val = tl.load(b_ptr + chan_idx)
        acc += bias_val

    # Store the result in the output tensor
    y_ptr_out = (
        y_ptr + 
        batch_idx * stride_y_b + 
        chan_idx * stride_y_c + 
        h_out_offsets[:, None] * stride_y_h + 
        w_out_offsets[None, :] * stride_y_w
    )
    tl.store(y_ptr_out, acc, mask=mask_h[:, None] & mask_w[None, :])


def triton_depthwise_conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    B, C, H, W = x.shape
    KH, KW = weight.shape[2], weight.shape[3]
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    # Calculate output dimensions
    H_out = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    W_out = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1

    y = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Strides
    stride_x_b, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_w_c, _, stride_w_kh, stride_w_kw = weight.stride()
    stride_y_b, stride_y_c, stride_y_h, stride_y_w = y.stride()

    # Kernel configuration
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 32
    
    grid = (B * C, triton.cdiv(H_out, BLOCK_SIZE_H), triton.cdiv(W_out, BLOCK_SIZE_W))

    depthwise_conv2d_kernel[grid](
        x, weight, bias if bias is not None else 0, y,
        B, C, H, W,
        KH, KW,
        H_out, W_out,
        sh, sw,
        ph, pw,
        dh, dw,
        stride_x_b, stride_x_c, stride_x_h, stride_x_w,
        stride_w_c, stride_w_kh, stride_w_kw,
        stride_y_b, stride_y_c, stride_y_h, stride_y_w,
        bias is not None,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)
        
        # Depthwise convolution: weight shape (in_channels, 1, KH, KW)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )