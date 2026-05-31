import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool_kernel(
    x_ptr, 
    out_ptr,
    N, C, H, W,
    H_out, W_out,
    stride, padding,
    K: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: (N * C, H_out, (W_out + BLOCK_W - 1) // BLOCK_W)
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Decompose pid_nc into batch and channel
    batch_id = pid_nc // C
    chan_id = pid_nc % C

    # Output width offsets for the current block
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    oh = pid_h

    # Calculate base pointers for the current batch and channel
    # Input: (N, C, H, W), Output: (N, C, H_out, W_out)
    x_base = x_ptr + batch_id * C * H * W + chan_id * H * W
    out_base = out_ptr + batch_id * C * H_out * W_out + chan_id * H_out * W_out + oh * W_out

    # Accumulator for the average pooling window
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Iterate over the pooling window (K x K)
    # Since K is a constexpr, these loops are unrolled by the Triton compiler
    for kh in range(K):
        h_in = oh * stride + kh - padding
        # Only process if the horizontal window offset is within the input image bounds
        if h_in >= 0 and h_in < H:
            row_ptr = x_base + h_in * W
            for kw in range(K):
                # Calculate input width index for each element in the BLOCK_W
                w_in = ow * stride + kw - padding
                # Mask to handle padding (out-of-bounds elements are treated as 0)
                mask = (w_in >= 0) & (w_in < W)
                val = tl.load(row_ptr + w_in, mask=mask, other=0.0)
                acc += val

    # Compute the average (PyTorch AvgPool2d defaults to count_include_pad=True)
    out = acc / (K * K)
    
    # Mask to ensure we don't write past the output tensor boundary
    mask_out = ow < W_out
    tl.store(out_base + ow, out, mask=mask_out)


def triton_avg_pool(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    Triton wrapper for 2D Average Pooling.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    N, C, H, W = x.shape
    s = stride if stride is not None else kernel_size
    p = padding

    # Calculate output dimensions
    h_out = (H + 2 * p - kernel_size) // s + 1
    w_out = (W + 2 * p - kernel_size) // s + 1

    # Prepare output tensor
    out = torch.empty((N, C, h_out, w_out), device=x.device, dtype=x.dtype)

    # Tuning parameter for the width block size
    BLOCK_W = 32
    
    # Grid: (N * C, H_out, W_out_blocks)
    grid = (N * C, h_out, (w_out + BLOCK_W - 1) // BLOCK_W)

    # Launch Triton kernel
    avg_pool_kernel[grid](
        x, out,
        N, C, H, W,
        h_out, w_out,
        s, p,
        K=kernel_size,
        BLOCK_W=BLOCK_W
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using the Triton implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool(x, self.kernel_size, self.stride, self.padding)