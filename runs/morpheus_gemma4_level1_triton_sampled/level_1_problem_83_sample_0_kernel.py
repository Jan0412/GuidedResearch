import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_asym_kernel(
    x_ptr, 
    w_ptr, 
    b_ptr, 
    out_ptr, 
    B, C, H, W, 
    KH, 
    stride, padding, dilation, 
    H_out, W_out, 
    S_B, S_C, S_H, S_W, 
    S_B_out, S_C_out, S_H_out, S_W_out, 
    S_C_w, S_KH_w, 
    has_bias: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    # Map program IDs to batch, channel, and width
    pid_0 = tl.program_id(0)  # B * C
    pid_1 = tl.program_id(1)  # W_out
    pid_2 = tl.program_id(2)  # H_out block

    b = pid_0 // C
    c = pid_0 % C
    w = pid_1
    
    # Height offsets for this block
    h_offsets = pid_2 * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask = h_offsets < H_out
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)
    
    # Loop over the asymmetric kernel height (KH, 1)
    for kh in range(KH):
        # Calculate input height index
        ih = h_offsets * stride - padding + kh * dilation
        ih_mask = (ih >= 0) & (ih < H)
        
        # Load input values: x[b, c, ih, w]
        # Pointer arithmetic: base + b*S_B + c*S_C + ih*S_H + w*S_W
        x_off = b * S_B + c * S_C + ih * S_H + w * S_W
        val = tl.load(x_ptr + x_off, mask=ih_mask, other=0.0)
        
        # Load weight value: w[c, 0, kh, 0]
        w_off = c * S_C_w + kh * S_KH_w
        weight_val = tl.load(w_ptr + w_off)
        
        acc += val * weight_val
        
    # Add bias if applicable
    if has_bias:
        bias_val = tl.load(b_ptr + c)
        acc += bias_val
        
    # Store output values: out[b, c, h, w]
    out_off = b * S_B_out + c * S_C_out + h_offsets * S_H_out + w * S_W_out
    tl.store(out_ptr + out_off, acc, mask=mask)

def triton_depthwise_conv2d_asym(x, weight, bias=None, stride=1, padding=0, dilation=1):
    # x: (B, C, H, W)
    # weight: (C, 1, KH, 1)
    B, C, H, W = x.shape
    KH = weight.shape[2]
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - 1) // stride + 1 # Since KW=1
    
    x = x.contiguous()
    weight = weight.contiguous()
    
    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Strides for x
    S_B, S_C, S_H, S_W = x.stride()
    # Strides for out
    S_B_out, S_C_out, S_H_out, S_W_out = out.stride()
    # Strides for weight (C, 1, KH, 1)
    S_C_w, _, S_KH_w, _ = weight.stride()
    
    BLOCK_SIZE_H = 32
    grid = (B * C, W_out, (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H)
    
    has_bias = bias is not None
    b_ptr = bias if has_bias else None
    
    depthwise_conv2d_asym_kernel[grid](
        x, weight, b_ptr, out,
        B, C, H, W,
        KH,
        stride, padding, dilation,
        H_out, W_out,
        S_B, S_C, S_H, S_W,
        S_B_out, S_C_out, S_H_out, S_W_out,
        S_C_w, S_KH_w,
        has_bias=has_bias,
        BLOCK_SIZE_H=BLOCK_SIZE_H
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution with an asymmetric kernel using Triton.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # Use nn.Conv2d to manage weights and bias
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
        # Use the Triton kernel for the forward pass
        return triton_depthwise_conv2d_asym(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )