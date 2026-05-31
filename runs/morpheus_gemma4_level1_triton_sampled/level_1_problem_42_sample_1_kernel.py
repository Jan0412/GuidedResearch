import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool2d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    ho,
    wo,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program IDs
    pid_bc = tl.program_id(0)  # batch * channels
    pid_h = tl.program_id(1)   # output height block
    pid_w = tl.program_id(2)   # output width block

    # Decode batch and channel
    batch_id = pid_bc // channels
    channel_id = pid_bc % channels

    # Output ranges for this block
    oh_range = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow_range = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # Masks for output boundaries
    mask_oh = oh_range < ho
    mask_ow = ow_range < wo

    # Compute base pointers for the current batch and channel
    # x_ptr is (batch, channel, height, width)
    x_base = x_ptr + batch_id * (channels * height * width) + channel_id * (height * width)
    # out_ptr is (batch, channel, ho, wo)
    out_base = out_ptr + batch_id * (channels * ho * wo) + channel_id * (ho * wo)

    # Initialize max values to -infinity
    max_val = tl.full((BLOCK_H, BLOCK_W), -float('inf'), dtype=tl.float32)

    # Iterate over the pooling window
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input coordinates
            # ih: (BLOCK_H, 1), iw: (1, BLOCK_W)
            ih = oh_range[:, None] * stride + kh * dilation - padding
            iw = ow_range[None, :] * stride + kw * dilation - padding

            # Boundary masks for input
            mask_ih = (ih >= 0) & (ih < height)
            mask_iw = (iw >= 0) & (iw < width)
            mask = mask_ih & mask_iw

            # Calculate offsets for the input tensor
            # offset: (BLOCK_H, BLOCK_W)
            offsets = ih * width + iw

            # Load values from input, using -inf for padded regions
            val = tl.load(x_base + offsets, mask=mask, other=-float('inf'))
            
            # Update max value
            max_val = tl.maximum(max_val, val)

    # Final output mask combining output boundary and window masks
    final_mask = mask_oh[:, None] & mask_ow[None, :]
    
    # Store result
    out_offsets = oh_range[:, None] * wo + ow_range[None, :]
    tl.store(out_base + out_offsets, max_val, mask=final_mask)


def triton_maxpool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, height, width = x.shape
    
    # Calculate output dimensions
    ho = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    wo = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty((batch_size, channels, ho, wo), device=x.device, dtype=x.dtype)
    
    # Tuning parameters
    BLOCK_H = 16
    BLOCK_W = 16
    
    # Grid definition
    grid = (
        batch_size * channels, 
        (ho + BLOCK_H - 1) // BLOCK_H, 
        (wo + BLOCK_W - 1) // BLOCK_W
    )
    
    maxpool2d_kernel[grid](
        x, out,
        batch_size, channels, height, width,
        ho, wo,
        kernel_size, stride, padding, dilation,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
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
        return triton_maxpool2d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )