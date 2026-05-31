import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, channels, h_in, w_in, h_out, w_out,
    kh, kw, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    has_bias,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Output offsets
    oh_offsets = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    ow_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Masks for output boundaries
    mask_h = oh_offsets < h_out
    mask_w = ow_offsets < w_out

    # Pointers to the start of the current batch and channel
    # x: (batch, channels, h_in, w_in)
    x_base = x_ptr + pid_n * (channels * h_in * w_in) + pid_c * (h_in * w_in)
    # w: (channels, 1, kh, kw)
    w_base = w_ptr + pid_c * (kh * kw)
    # out: (batch, channels, h_out, w_out)
    out_base = out_ptr + pid_n * (channels * h_out * w_out) + pid_c * (h_out * w_out)

    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    for i in range(kh):
        for j in range(kw):
            # Input indices for the current kernel element (i, j)
            ih_offsets = oh_offsets * stride_h + i * dil_h - pad_h
            iw_offsets = ow_offsets * stride_w + j * dil_w - pad_w
            
            # Load weight for the current channel and kernel position
            weight = tl.load(w_base + i * kw + j)
            
            # Expand offsets to 2D for block loading
            ih_expanded = ih_offsets[:, None]
            iw_expanded = iw_offsets[None, :]
            
            # Compute linear offsets for x
            x_offsets = ih_expanded * w_in + iw_expanded
            
            # Boundary mask for input
            mask_ih = (ih_expanded >= 0) & (ih_expanded < h_in)
            mask_iw = (iw_expanded >= 0) & (iw_expanded < w_in)
            mask_x = mask_ih & mask_iw
            
            # Load x patch and accumulate
            x_vals = tl.load(x_base + x_offsets, mask=mask_x, other=0.0)
            acc += x_vals * weight

    # Add bias if applicable
    if has_bias:
        bias_val = tl.load(b_ptr + pid_c)
        acc += bias_val

    # Store results back to output tensor
    out_offsets = oh_offsets[:, None] * w_out + ow_offsets[None, :]
    tl.store(out_base + out_offsets, acc, mask=mask_h[:, None] & mask_w[None, :])


def triton_depthwise_conv(x, conv_layer):
    # Extract weights and parameters from the Conv2d layer
    w = conv_layer.weight
    b = conv_layer.bias
    
    # Ensure tensors are contiguous on GPU
    x = x.contiguous()
    w = w.contiguous()
    
    batch, channels, h_in, w_in = x.shape
    kh, kw = w.shape[2], w.shape[3]
    sh, sw = conv_layer.stride
    ph, pw = conv_layer.padding
    dh, dw = conv_layer.dilation
    
    # Calculate output dimensions
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    
    out = torch.empty((batch, channels, h_out, w_out), device=x.device, dtype=x.dtype)
    
    # Bias handling
    has_bias = 1 if b is not None else 0
    b_ptr = b.contiguous() if b is not None else torch.zeros(1, device=x.device, dtype=x.dtype)
    
    # Triton block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    
    # Grid: (batch, channels, output_h_blocks, output_w_blocks)
    grid = (
        batch, 
        channels, 
        (h_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, 
        (w_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )
    
    depthwise_conv_kernel[grid](
        x, w, b_ptr, out,
        batch, channels, h_in, w_in, h_out, w_out,
        kh, kw, sh, sw, ph, pw, dh, dw,
        has_bias,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using a custom Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the Conv2d layer to manage weights and bias initialization
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, (kernel_size_h, kernel_size_w), 
            stride=(stride_h, stride_w), padding=(padding_h, padding_w), 
            dilation=(dilation_h, dilation_w), groups=in_channels, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using the Triton-optimized kernel.
        """
        return triton_depthwise_conv(x, self.conv2d)