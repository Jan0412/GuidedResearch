import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C, H, W, KH, OH, OW,
    S, P, D,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wkh,
    stride_ob, stride_oc, stride_oh, stride_ow,
    has_bias,
    BLOCK_OH: tl.constexpr,
):
    # pid_0 maps to (batch, channel, output_width)
    pid_0 = tl.program_id(0)
    # pid_1 maps to the block of output_height
    pid_1 = tl.program_id(1)

    # Decompose pid_0 into b, c, ow
    b = pid_0 // (C * OW)
    rem = pid_0 % (C * OW)
    c = rem // OW
    ow = rem % OW

    # Range of output height indices for this block
    oh_offsets = pid_1 * BLOCK_OH + tl.arange(0, BLOCK_OH)
    mask_oh = oh_offsets < OH

    # Input width index calculation
    # Since kernel width is 1, the window is just a single pixel
    iw = ow * S - P
    mask_w = (iw >= 0) & (iw < W)

    # Pointers to weight and bias for the current channel
    w_chan_ptr = w_ptr + c * stride_wc
    
    # Output pointer base for the current (b, c, ow)
    out_base_ptr = out_ptr + b * stride_ob + c * stride_oc + ow * stride_ow

    # Initialize accumulator for the output height block
    acc = tl.zeros([BLOCK_OH], dtype=tl.float32)

    # Loop over the kernel height
    for kh in range(KH):
        # Input height index calculation
        ih = oh_offsets * S - P + kh * D
        mask_h = (ih >= 0) & (ih < H)
        mask = mask_oh & mask_h & mask_w

        # Load input: x[b, c, ih, iw]
        x_ptr_val = x_ptr + b * stride_xb + c * stride_xc + ih * stride_xh + iw * stride_xw
        x_val = tl.load(x_ptr_val, mask=mask, other=0.0)

        # Load weight: w[c, 0, kh, 0]
        w_val = tl.load(w_chan_ptr + kh * stride_wkh)

        acc += x_val * w_val

    # Add bias if applicable
    if has_bias:
        bias_val = tl.load(b_ptr + c)
        acc += bias_val

    # Store the result in output: out[b, c, oh, ow]
    tl.store(out_base_ptr + oh_offsets * stride_oh, acc, mask=mask_oh)


def triton_depthwise_conv2d(x, weight, bias, stride, padding, dilation):
    # Ensure inputs are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C, H, W = x.shape
    _, _, KH, _ = weight.shape
    S, P, D = stride, padding, dilation
    
    # Calculate output dimensions
    OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
    OW = (W + 2 * P - 1) // S + 1
    
    out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
    
    # Get strides
    stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wc, _, stride_wkh, _ = weight.stride()
    stride_ob, stride_oc, stride_oh, stride_ow = out.stride()
    
    has_bias = bias is not None
    b_ptr = bias.data_ptr() if has_bias else 0
    
    BLOCK_OH = 32
    grid = (B * C * OW, triton.cdiv(OH, BLOCK_OH))
    
    depthwise_conv_kernel[grid](
        x, weight, b_ptr, out,
        B, C, H, W, KH, OH, OW,
        S, P, D,
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_wc, stride_wkh,
        stride_ob, stride_oc, stride_oh, stride_ow,
        has_bias,
        BLOCK_OH=BLOCK_OH,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution with an asymmetric kernel using Triton.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the Conv2d layer to manage weights and bias
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
        # Extract parameters from the Conv2d layer
        weight = self.conv2d.weight
        bias = self.conv2d.bias
        stride = self.conv2d.stride[0]
        padding = self.conv2d.padding[0]
        dilation = self.conv2d.dilation[0]
        
        # Call the Triton-optimized kernel
        return triton_depthwise_conv2d(x, weight, bias, stride, padding, dilation)