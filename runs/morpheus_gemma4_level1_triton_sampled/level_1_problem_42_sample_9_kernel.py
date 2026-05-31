import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    OH, OW,
    stride, padding, dilation,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for (batch * channel * oh)
    pid_0 = tl.program_id(0)
    # Program ID for ow block
    pid_1 = tl.program_id(1)

    # Decompose pid_0 into b, c, oh
    oh = pid_0 % OH
    temp = pid_0 // OH
    c = temp % C
    b = temp // C

    # ow offsets for this block
    ow_offsets = pid_1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = ow_offsets < OW

    # Initialize max_val to -infinity
    max_val = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)

    # Loop over the kernel window
    for kh in range(KERNEL_SIZE):
        ih = oh * stride + kh * dilation - padding
        if ih < 0 or ih >= H:
            continue
        
        # Base pointer for the current row in the kernel window
        row_ptr = x_ptr + b * stride_x_b + c * stride_x_c + ih * stride_x_h

        for kw in range(KERNEL_SIZE):
            iw = ow_offsets * stride + kw * dilation - padding
            # Bounds check for iw
            iw_mask = (iw >= 0) & (iw < W)
            
            # Calculate input pointer for the current element in the kernel window
            curr_x_ptr = row_ptr + iw * stride_x_w
            
            # Load and update max
            val = tl.load(curr_x_ptr, mask=iw_mask, other=-float('inf'))
            max_val = tl.maximum(max_val, val)

    # Store result
    out_ptr_base = out_ptr + b * stride_out_b + c * stride_out_c + oh * stride_out_h + ow_offsets * stride_out_w
    tl.store(out_ptr_base, max_val, mask=out_mask)


def triton_maxpool(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    N, C, H, W = x.shape

    # Calculate output dimensions
    OH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    # Strides for input and output tensors
    stride_x_b, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_out_b, stride_out_c, stride_out_h, stride_out_w = out.stride()

    BLOCK_SIZE = 128
    # Grid: (N * C * OH) programs for the height/channel/batch, 
    # and (OW / BLOCK_SIZE) programs for the width block.
    grid = (N * C * OH, triton.cdiv(OW, BLOCK_SIZE))

    maxpool_kernel[grid](
        x, out,
        N, C, H, W,
        OH, OW,
        stride, padding, dilation,
        stride_x_b, stride_x_c, stride_x_h, stride_x_w,
        stride_out_b, stride_out_c, stride_out_h, stride_out_w,
        KERNEL_SIZE=kernel_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using custom Triton kernels.
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
        return triton_maxpool(x, self.kernel_size, self.stride, self.padding, self.dilation)