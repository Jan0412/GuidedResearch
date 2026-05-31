import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool_kernel(
    x_ptr,
    out_ptr,
    H,
    W,
    OH,
    OW,
    S: tl.constexpr,
    P: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_bc = tl.program_id(0)  # Batch * Channels
    pid_h = tl.program_id(1)   # Output Height block
    pid_w = tl.program_id(2)   # Output Width block

    # Output offsets
    oh_offsets = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    ow_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Reshape to 2D for broadcasting
    oh = oh_offsets[:, None]
    ow = ow_offsets[None, :]

    # Mask for output boundaries
    mask_out = (oh < OH) & (ow < OW)

    # Base pointers for the current batch/channel
    x_base = x_ptr + pid_bc * (H * W)
    out_base = out_ptr + pid_bc * (OH * OW)

    # Accumulator for the sum
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Iterate over the pooling window
    for kh in range(K):
        for kw in range(K):
            # Calculate input coordinates
            h_idx = oh * S + kh - P
            w_idx = ow * S + kw - P
            
            # Mask for input boundaries (handling padding)
            mask_in = (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)
            
            # Calculate offset in the input tensor
            # offset = h_idx * W + w_idx
            # We use broadcasting: (BLOCK_SIZE_H, 1) * W + (1, BLOCK_SIZE_W)
            offset = h_idx * W + w_idx
            
            # Load and accumulate
            val = tl.load(x_base + offset, mask=mask_in, other=0.0)
            acc += val

    # Compute average and store
    # nn.AvgPool2d default: count_include_pad=True, so divide by K*K
    res = acc / (K * K)
    tl.store(out_base + oh * OW + ow, res, mask=mask_out)


def triton_avg_pool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    # Input shapes
    N, C, H, W = x.shape
    
    # Handle stride=None (default to kernel_size)
    S = stride if stride is not None else kernel_size
    P = padding
    K = kernel_size
    
    # Calculate output dimensions
    OH = (H + 2 * P - K) // S + 1
    OW = (W + 2 * P - K) // S + 1
    
    # Prepare output tensor
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    
    # Tuning parameters
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    
    # Grid: (Batch * Channels, Output Height Blocks, Output Width Blocks)
    grid = (N * C, triton.cdiv(OH, BLOCK_SIZE_H), triton.cdiv(OW, BLOCK_SIZE_W))
    
    # Launch kernel
    avg_pool_kernel[grid](
        x, out,
        H, W, OH, OW,
        S=S, P=P, K=K,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using a custom Triton kernel.
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
        Applies 2D Average Pooling to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        # Ensure input is contiguous on GPU
        x = x.contiguous()
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)