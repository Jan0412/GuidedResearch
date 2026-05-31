import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C, H, W, KH, KW, S, P,
    H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wk, stride_ww,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    # pid_0: batch * C * H_out
    # pid_1: W_out block
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    
    # Unpack pid_0 to get batch, channel, and output height
    b = pid_0 // (C * H_out)
    rem = pid_0 % (C * H_out)
    c = rem // H_out
    oh = rem % H_out
    
    # Output width offsets for the current block
    ow = pid_1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_ow = ow < W_out
    
    # Base pointers for the current batch and channel
    x_base = x_ptr + b * stride_xb + c * stride_xc
    w_base = w_ptr + c * stride_wc
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over the kernel dimensions
    for kh in range(KH):
        ih = oh * S + kh - P
        if ih < 0 or ih >= H:
            continue
        
        # Pointer to the specific row in the input tensor
        x_row_ptr = x_base + ih * stride_xh
        
        for kw in range(KW):
            # Load the weight for this specific depthwise kernel position
            weight = tl.load(w_base + kh * stride_wk + kw * stride_ww)
            
            # Calculate input width indices and handle padding/bounds
            iw = ow * S + kw - P
            mask_iw = (iw >= 0) & (iw < W) & mask_ow
            
            # Load input values and accumulate
            val = tl.load(x_row_ptr + iw * stride_xw, mask=mask_iw, other=0.0)
            acc += val * weight
    
    # Add bias if it exists
    if b_ptr is not None:
        bias = tl.load(b_ptr + c)
        acc += bias
        
    # Store the final computed values into the output tensor
    out_base = out_ptr + b * stride_ob + c * stride_oc + oh * stride_oh
    tl.store(out_base + ow * stride_ow, acc, mask=mask_ow)

def triton_depthwise_conv(x, weight, bias, stride, padding):
    # Ensure tensors are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, C, H, W = x.shape
    KH, _, KW, _ = weight.shape # weight is (C, 1, KH, KW) for depthwise
    # Correcting weight shape access for nn.Conv2d depthwise: (out_channels, in_channels/groups, kH, kW)
    # For depthwise, out_channels = C, in_channels/groups = 1.
    KH, KW = weight.shape[2], weight.shape[3]
    
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1
    
    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Element strides
    stride_xb, stride_xc, stride_xh, stride_xw = C * H * W, H * W, W, 1
    stride_wc, stride_wk, stride_ww = 1 * KH * KW, KW, 1
    stride_ob, stride_oc, stride_oh, stride_ow = C * H_out * W_out, H_out * W_out, W_out, 1
    
    BLOCK_SIZE = 32
    grid = (B * C * H_out, triton.cdiv(W_out, BLOCK_SIZE))
    
    depthwise_conv_kernel[grid](
        x, weight, bias, out,
        B, C, H, W, KH, KW, stride, padding,
        H_out, W_out,
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_wc, stride_wk, stride_ww,
        stride_ob, stride_oc, stride_oh, stride_ow,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        # Use nn.Conv2d to manage parameters (weight and bias)
        self.conv2d = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels, 
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the Triton-optimized depthwise convolution
        return triton_depthwise_conv(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.stride, 
            self.padding
        )