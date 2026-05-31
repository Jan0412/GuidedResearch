import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool2d_kernel(
    x_ptr,
    out_ptr,
    B, C, H, W,
    kH, kW,
    sH, sW,
    pH, pW,
    dH, dW,
    oH, oW,
    stride_n, stride_c,
    out_stride_n, out_stride_c,
    BLOCK_OW: tl.constexpr,
):
    # Program ID mapping
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # Map pid_0 to (batch, channel, out_h)
    # pid_0 = n * (C * oH) + c * oH + oh
    oh = pid_0 % oH
    bc = pid_0 // oH
    n = bc // C
    c = bc % C

    # Map pid_1 to a block of out_w
    ow_start = pid_1 * BLOCK_OW
    ow_offsets = ow_start + tl.arange(0, BLOCK_OW)
    mask_ow = ow_offsets < oW

    # Pointers to the start of the current (n, c) slice
    x_base_ptr = x_ptr + n * stride_n + c * stride_c
    out_base_ptr = out_ptr + n * out_stride_n + c * out_stride_c + oh * oW

    # Window offsets
    # h_indices: (kH,)
    h_indices = (oh * sH - pH + tl.arange(0, kH) * dH)
    # w_indices: (BLOCK_OW, kW)
    w_start = ow_offsets * sW - pW
    w_indices = w_start[:, None] + tl.arange(0, kW)[None, :] * dW

    # Create 3D offset tensor: (kH, BLOCK_OW, kW)
    # x_base_ptr + h * W + w
    h_expanded = h_indices[:, None, None]
    w_expanded = w_indices[None, :, :]
    offsets = h_expanded * W + w_expanded

    # Masks for boundary checks
    mask_h = (h_expanded >= 0) & (h_expanded < H)
    mask_w = (w_expanded >= 0) & (w_expanded < W)
    mask = mask_h & mask_w

    # Load the window values
    # Shape: (kH, BLOCK_OW, kW)
    vals = tl.load(x_base_ptr + offsets, mask=mask, other=-float('inf'))

    # Max pooling over the window (kH and kW dimensions)
    # axis=0 reduces kH -> (BLOCK_OW, kW)
    res = tl.max(vals, axis=0)
    # axis=1 reduces kW -> (BLOCK_OW,)
    res = tl.max(res, axis=1)

    # Store the result
    tl.store(out_base_ptr + ow_offsets, res, mask=mask_ow)


def triton_maxpool2d(x, kernel_size, stride, padding, dilation):
    # Input shape: (B, C, H, W)
    B, C, H, W = x.shape
    
    # For simplicity, we assume kernel_size, stride, etc. are integers
    kH = kW = kernel_size
    sH = sW = stride
    pH = pW = padding
    dH = dW = dilation

    # Calculate output dimensions
    oH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    oW = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    out = torch.empty((B, C, oH, oW), device=x.device, dtype=x.dtype)

    # Strides
    stride_n = C * H * W
    stride_c = H * W
    out_stride_n = C * oH * oW
    out_stride_c = oH * oW

    BLOCK_OW = 16
    grid = (B * C * oH, (oW + BLOCK_OW - 1) // BLOCK_OW)

    maxpool2d_kernel[grid](
        x, out,
        B, C, H, W,
        kH, kW,
        sH, sW,
        pH, pW,
        dH, dW,
        oH, oW,
        stride_n, stride_c,
        out_stride_n, out_stride_c,
        BLOCK_OW=BLOCK_OW,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        # Ensure input is contiguous and on GPU
        x = x.contiguous()
        return triton_maxpool2d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )