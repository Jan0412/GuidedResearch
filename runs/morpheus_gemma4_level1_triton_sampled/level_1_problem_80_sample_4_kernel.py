import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    x_sB, x_sC, x_sH, x_sW,
    w_sOC, w_sIC, w_sKH, w_sKW,
    out_sB, out_sOC, out_sOH, out_sOW,
    B, C_in, H, W, C_out, Kh, Kw,
    S, Ph, Pw, Dh, Dw,
    Oh, Ow,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    batch_id = tl.program_id(0)
    oc_id = tl.program_id(1)
    oh_id = tl.program_id(2)
    ow_group_id = tl.program_id(3)

    # Output width offsets
    ow_offsets = ow_group_id * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    mask_ow = ow_offsets < Ow

    # Initialize accumulator with bias if it exists
    # b_ptr is None if bias is False, but in Triton we pass a dummy or check
    acc = 0.0
    if b_ptr != 0:
        acc = tl.load(b_ptr + oc_id)

    # Loop over input channels and kernel dimensions
    for ic in range(C_in):
        for kh in range(Kh):
            # Calculate input height index
            ih = oh_id * S + kh * Dh - Ph
            if ih < 0 or ih >= H:
                continue
            
            for kw in range(Kw):
                # Calculate input width indices (vectorized)
                iw = ow_offsets * S + kw * Dw - Pw
                
                # Mask for input width bounds
                mask_iw = (iw >= 0) & (iw < W) & mask_ow
                
                # Load weight (scalar for this block of output width)
                w_val = tl.load(w_ptr + oc_id * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW)
                
                # Load input (vector)
                x_ptr_off = x_ptr + batch_id * x_sB + ic * x_sC + ih * x_sH + iw * x_sW
                x_val = tl.load(x_ptr_off, mask=mask_iw, other=0.0)
                
                acc += x_val * w_val

    # Store result
    out_ptr_off = out_ptr + batch_id * out_sB + oc_id * out_sOC + oh_id * out_sOH + ow_offsets * out_sOW
    tl.store(out_ptr_off, acc, mask=mask_ow)


def triton_conv2d(x, w, b, stride, padding, dilation):
    # x: (B, C_in, H, W)
    # w: (C_out, C_in, Kh, Kw)
    # b: (C_out,) or None
    B, C_in, H, W = x.shape
    C_out, _, Kh, Kw = w.shape
    S = stride
    Ph, Pw = padding
    Dh, Dw = dilation

    # Calculate output dimensions
    Oh = (H + 2 * Ph - Dh * (Kh - 1) - 1) // S + 1
    Ow = (W + 2 * Pw - Dw * (Kw - 1) - 1) // S + 1

    out = torch.empty((B, C_out, Oh, Ow), device=x.device, dtype=x.dtype)

    # Ensure tensors are contiguous
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()

    # Strides
    x_sB, x_sC, x_sH, x_sW = x.stride()
    w_sOC, w_sIC, w_sKH, w_sKW = w.stride()
    out_sB, out_sOC, out_sOH, out_sOW = out.stride()

    BLOCK_SIZE_W = 32
    grid = (B, C_out, Oh, (Ow + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

    conv2d_kernel[grid](
        x, w, b if b is not None else 0, out,
        x_sB, x_sC, x_sH, x_sW,
        w_sOC, w_sIC, w_sKH, w_sKW,
        out_sB, out_sOC, out_sOH, out_sOW,
        B, C_in, H, W, C_out, Kh, Kw,
        S, Ph, Pw, Dh, Dw,
        Oh, Ow,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        # We use nn.Conv2d to handle parameter initialization and storage
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract weights and bias from the nn.Conv2d layer
        w = self.conv_layer.weight
        b = self.conv_layer.bias
        
        # Call the Triton implementation
        return triton_conv2d(x, w, b, self.stride, self.padding, self.dilation)