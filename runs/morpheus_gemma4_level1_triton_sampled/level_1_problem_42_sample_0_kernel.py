import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool_kernel(
    x_ptr, 
    out_ptr,
    batch, 
    channels, 
    h, 
    w,
    oh, 
    ow,
    kh, 
    kw, 
    sh, 
    sw, 
    ph, 
    pw, 
    dh, 
    dw,
    stride_x_b, 
    stride_x_c, 
    stride_x_h, 
    stride_x_w,
    stride_out_b, 
    stride_out_c, 
    stride_out_h, 
    stride_out_w,
    BLOCK_SIZE_H: tl.constexpr, 
    BLOCK_SIZE_W: tl.constexpr
):
    # Program IDs
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Compute batch and channel indices
    b = pid_bc // channels
    c = pid_bc % channels

    # Compute output coordinate ranges for this block
    oh_offsets = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    ow_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Masks for output boundaries
    mask_oh = oh_offsets < oh
    mask_ow = ow_offsets < ow
    block_mask = mask_oh[:, None] & mask_ow[None, :]

    # Base pointers for the current batch and channel
    base_x_ptr = x_ptr + b * stride_x_b + c * stride_x_c
    base_out_ptr = out_ptr + b * stride_out_b + c * stride_out_c

    # Initialize max values to negative infinity
    max_val = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W), float("-inf"), dtype=tl.float32)

    # Iterate over the kernel window
    for i in range(kh):
        for j in range(kw):
            # Calculate input coordinates for the current kernel element
            # Formula: input_coord = output_coord * stride + kernel_idx * dilation - padding
            ih = oh_offsets[:, None] * sh + i * dh - ph
            iw = ow_offsets[None, :] * sw + j * dw - pw

            # Mask for input boundaries (padding)
            mask_in = (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
            
            # Load values from input tensor
            # Pointer arithmetic: base + ih * stride_h + iw * stride_w
            ptr = base_x_ptr + ih * stride_x_h + iw * stride_x_w
            val = tl.load(ptr, mask=mask_in & block_mask, other=float("-inf"))
            
            # Update max
            max_val = tl.maximum(max_val, val)

    # Store the result
    out_ptr_block = base_out_ptr + oh_offsets[:, None] * stride_out_h + ow_offsets[None, :] * stride_out_w
    tl.store(out_ptr_block, max_val, mask=block_mask)


def triton_maxpool2d(x, kernel_size, stride, padding, dilation):
    # Ensure input is contiguous and on GPU
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    
    # Handle cases where parameters might be tuples
    kh = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
    kw = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size
    sh = stride[0] if isinstance(stride, tuple) else stride
    sw = stride[1] if isinstance(stride, tuple) else stride
    ph = padding[0] if isinstance(padding, tuple) else padding
    pw = padding[1] if isinstance(padding, tuple) else padding
    dh = dilation[0] if isinstance(dilation, tuple) else dilation
    dw = dilation[1] if isinstance(dilation, tuple) else dilation

    batch, channels, h, w = x.shape
    
    # Calculate output dimensions
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    
    out = torch.empty((batch, channels, oh, ow), device=x.device, dtype=x.dtype)

    # Strides for NCHW layout
    stride_x_b, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_out_b, stride_out_c, stride_out_h, stride_out_w = out.stride()

    # Tuning parameters
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    # Grid: (batch * channels, ceil(oh / BLOCK_SIZE_H), ceil(ow / BLOCK_SIZE_W))
    grid = (
        batch * channels, 
        triton.cdiv(oh, BLOCK_SIZE_H), 
        triton.cdiv(ow, BLOCK_SIZE_W)
    )

    maxpool_kernel[grid](
        x, out,
        batch, channels, h, w,
        oh, ow,
        kh, kw, sh, sw, ph, pw, dh, dw,
        stride_x_b, stride_x_c, stride_x_h, stride_x_w,
        stride_out_b, stride_out_c, stride_out_h, stride_out_w,
        BLOCK_SIZE_H=BLOCK_SIZE_H, 
        BLOCK_SIZE_W=BLOCK_SIZE_W
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
        Applies Max Pooling 2D to the input tensor using the Triton implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        # Ensure input is FP32 for the kernel
        x = x.to(torch.float32)
        return triton_maxpool2d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )