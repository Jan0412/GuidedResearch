import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, H_in, W_in,
    C_out, KH, KW,
    S, P,
    H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wb, stride_wc, stride_wh, stride_ww,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Decompose pid_bc into batch index and output channel index
    b = pid_bc // C_out
    oc = pid_bc % C_out
    
    # In depthwise convolution with groups=C_in, 
    # the input channel corresponding to output channel oc is:
    multiplier = C_out // C_in
    ic = oc // multiplier

    # Width offsets for the block
    w_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    mask_w = w_offsets < W_out

    # Calculate output pointer for this block
    out_offset = (b * stride_ob + 
                  oc * stride_oc + 
                  pid_h * stride_oh + 
                  w_offsets * stride_ow)
    
    acc = tl.zeros([BLOCK_SIZE_W], dtype=tl.float32)

    # Loop over kernel dimensions
    for kh in range(KH):
        for kw in range(KW):
            # Calculate input coordinates
            h_in = pid_h * S + kh - P
            w_in = w_offsets * S + kw - P
            
            # Boundary masks for padding
            mask_h = (h_in >= 0) & (h_in < H_in)
            mask_w_pad = (w_in >= 0) & (w_in < W_in) & mask_w
            
            # Combine masks
            load_mask = mask_h & mask_w_pad
            
            # Compute input offset: x[b, ic, h_in, w_in]
            x_offset = (b * stride_xb + 
                        ic * stride_xc + 
                        h_in * stride_xh + 
                        w_in * stride_xw)
            
            # Load input value
            x_val = tl.load(x_ptr + x_offset, mask=load_mask, other=0.0)
            
            # Compute weight offset: w[oc, 0, kh, kw]
            w_offset = (oc * stride_wb + 
                        0 * stride_wc + 
                        kh * stride_wh + 
                        kw * stride_ww)
            
            # Load weight value
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate product
            acc += x_val * w_val

    # Add bias if it exists
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + oc)
        acc += bias_val

    # Store result
    tl.store(out_ptr + out_offset, acc, mask=mask_w)


def triton_depthwise_conv2d(x, weight, bias, stride, padding):
    # Input dimensions
    B, C_in, H_in, W_in = x.shape
    C_out, _, KH, KW = weight.shape
    
    # Output dimensions
    H_out = (H_in + 2 * padding - KH) // stride + 1
    W_out = (W_in + 2 * padding - KW) // stride + 1
    
    # Ensure tensors are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    out = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Get strides
    stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wb, stride_wc, stride_wh, stride_ww = weight.stride()
    stride_ob, stride_oc, stride_oh, stride_ow = out.stride()
    
    BLOCK_SIZE_W = 128
    
    # Grid: (Batch * OutChannels, OutHeight, OutWidth / BLOCK_SIZE_W)
    grid = (B * C_out, H_out, triton.cdiv(W_out, BLOCK_SIZE_W))
    
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, H_in, W_in,
        C_out, KH, KW,
        stride, padding,
        H_out, W_out,
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_wb, stride_wc, stride_wh, stride_ww,
        stride_ob, stride_oc, stride_oh, stride_ow,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using a custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        # We keep the nn.Conv2d to manage the parameters (weights and bias)
        self.conv2d = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=(kernel_size, kernel_size), 
            stride=stride, 
            padding=padding, 
            groups=in_channels, 
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using the Triton kernel.
        """
        return triton_depthwise_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.stride, 
            self.padding
        )