import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C, H, W,
    H_out, W_out,
    S_B_x, S_C_x, S_H_x, S_W_x,
    S_C_w, S_K_w,
    S_B_out, S_C_out, S_H_out, S_W_out,
    s, p, d,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # pid_0 represents the combination of (batch, channel, width_out)
    pid_0 = tl.program_id(0)
    # pid_1 represents the block of height_out
    pid_1 = tl.program_id(1)

    # Decompose pid_0
    b = pid_0 // (C * W_out)
    rem = pid_0 % (C * W_out)
    c = rem // W_out
    w_out = rem % W_out

    # Height offsets for this block
    h_start = pid_1 * BLOCK_SIZE
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE)
    h_mask = h_offsets < H_out

    # Load weights for the current channel
    # Weight shape is (C, 1, K, 1)
    w_offsets = tl.arange(0, K)
    # weight_ptr = w_ptr + c * S_C_w + 0 * S_1 + w_offsets * S_K_w + 0 * S_1
    weights = tl.load(w_ptr + c * S_C_w + w_offsets * S_K_w, mask=w_offsets < K, other=0.0)

    # Perform convolution along the height dimension
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(K):
        # Calculate input height index: h_in = h_out * stride - padding + k * dilation
        h_in = h_offsets * s - p + k * d
        # Mask for valid input boundaries
        in_mask = h_mask & (h_in >= 0) & (h_in < H)
        
        # Load input: x[b, c, h_in, w_out]
        x_ptr_val = x_ptr + b * S_B_x + c * S_C_x + h_in * S_H_x + w_out * S_W_x
        val = tl.load(x_ptr_val, mask=in_mask, other=0.0)
        
        # Multiply and accumulate
        acc += val * weights[k]

    # Add bias if provided
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + c)
        acc += bias_val

    # Store result: out[b, c, h_out, w_out]
    out_ptr_val = out_ptr + b * S_B_out + c * S_C_out + h_offsets * S_H_out + w_out * S_W_out
    tl.store(out_ptr_val, acc, mask=h_mask)


def triton_depthwise_conv2d(x, weight, bias, s, p, d):
    # Input shapes
    B, C, H, W = x.shape
    K = weight.shape[2] # Weight shape (C, 1, K, 1)
    
    # Calculate output dimensions
    H_out = (H + 2 * p - d * (K - 1) - 1) // s + 1
    W_out = (W + 2 * p - d * (1 - 1) - 1) // s + 1
    
    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Get strides
    S_B_x, S_C_x, S_H_x, S_W_x = x.stride()
    S_C_w, S_1_w, S_K_w, S_1_w_2 = weight.stride()
    S_B_out, S_C_out, S_H_out, S_W_out = out.stride()
    
    BLOCK_SIZE = 128
    # Grid: (B * C * W_out) programs for the spatial/channel dimensions,
    # and ceil(H_out / BLOCK_SIZE) programs for the height dimension.
    grid = (B * C * W_out, triton.cdiv(H_out, BLOCK_SIZE))
    
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        B, C, H, W,
        H_out, W_out,
        S_B_x, S_C_x, S_H_x, S_W_x,
        S_C_w, S_K_w,
        S_B_out, S_C_out, S_H_out, S_W_out,
        s, p, d,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using a custom Triton kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # We use nn.Conv2d to manage the parameters (weight and bias)
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
        # Ensure inputs are on GPU and contiguous for the Triton kernel
        x = x.contiguous()
        weight = self.conv2d.weight.contiguous()
        bias = self.conv2d.bias.contiguous() if self.conv2d.bias is not None else None
        
        return triton_depthwise_conv2d(
            x, 
            weight, 
            bias, 
            self.stride, 
            self.padding, 
            self.dilation
        )