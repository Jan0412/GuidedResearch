import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool_kernel(
    x_ptr,
    out_ptr,
    stride_c,
    stride_h,
    stride_w,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    H,
    W,
    OH,
    OW,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_nc = tl.program_id(0)  # Combined Batch and Channel index
    pid_h = tl.program_id(1)   # Output height block index
    pid_w = tl.program_id(2)   # Output width block index

    # Calculate the range of output pixels this block handles
    oh_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # Masks for output boundaries
    mask_oh = oh_offsets < OH
    mask_ow = ow_offsets < OW

    # Pointers to the start of the current channel's input and output maps
    # pid_nc = batch_idx * channels + channel_idx
    x_base_ptr = x_ptr + pid_nc * stride_c
    out_base_ptr = out_ptr + pid_nc * out_stride_c

    # Accumulator for the sum of the pooling window
    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    # Iterate over the pooling window
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input coordinates for the current window offset
            ih = oh_offsets * stride - padding + kh
            iw = ow_offsets * stride - padding + kw

            # Boundary masks for the input tensor (handling padding)
            mask_ih = (ih >= 0) & (ih < H)
            mask_iw = (iw >= 0) & (iw < W)
            
            # Combined mask for the current load: output boundaries AND input boundaries
            # Shape: [BLOCK_H, BLOCK_W]
            mask = mask_oh[:, None] & mask_ow[None, :] & mask_ih[:, None] & mask_iw[None, :]

            # Calculate offsets for the current window element across the block
            # ih is [BLOCK_H], iw is [BLOCK_W]
            offsets = ih[:, None] * stride_h + iw[None, :] * stride_w
            
            # Load values and accumulate
            val = tl.load(x_base_ptr + offsets, mask=mask, other=0.0)
            acc += val

    # Calculate average (PyTorch AvgPool2d default: count_include_pad=True)
    res = acc / (kernel_size * kernel_size)

    # Calculate output offsets for storing the result
    out_offsets = oh_offsets[:, None] * out_stride_h + ow_offsets[None, :] * out_stride_w
    
    # Store the result
    tl.store(out_base_ptr + out_offsets, res, mask=mask_oh[:, None] & mask_ow[None, :])


def triton_avg_pool(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    # Ensure input is contiguous
    x = x.contiguous()
    N, C, H, W = x.shape
    
    if stride is None:
        stride = kernel_size

    # Calculate output dimensions
    OH = (H + 2 * padding - kernel_size) // stride + 1
    OW = (W + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    
    # Strides for the input and output tensors
    stride_c = H * W
    stride_h = W
    stride_w = 1
    
    out_stride_c = OH * OW
    out_stride_h = OW
    out_stride_w = 1

    # Tuning parameters
    BLOCK_H = 16
    BLOCK_W = 16

    # Grid: (Batch * Channels, OH blocks, OW blocks)
    grid = (N * C, (OH + BLOCK_H - 1) // BLOCK_H, (OW + BLOCK_W - 1) // BLOCK_W)

    avg_pool_kernel[grid](
        x, out,
        stride_c, stride_h, stride_w,
        out_stride_c, out_stride_h, out_stride_w,
        H, W, OH, OW,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using a custom Triton kernel.
        """
        return triton_avg_pool(x, self.kernel_size, self.stride, self.padding)