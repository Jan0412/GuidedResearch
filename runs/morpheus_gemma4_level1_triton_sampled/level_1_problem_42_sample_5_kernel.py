import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool_kernel(
    x_ptr, out_ptr,
    B, C, H, W,
    OH, OW,
    k, s, p, d,
    BLOCK_K: tl.constexpr,
):
    # Map program IDs to output coordinates
    # pid_0 represents the combined (batch, channel, output_height)
    # pid_1 represents the output_width
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    b = pid_0 // (C * OH)
    rem = pid_0 % (C * OH)
    c = rem // OH
    oh = rem % OH
    ow = pid_1

    # Base offset for the current batch and channel in the input tensor (B, C, H, W)
    offset_base = b * C * H * W + c * H * W
    
    # Flat offset for the current output element (B, C, OH, OW)
    out_offset = b * C * OH * OW + c * OH * OW + oh * OW + ow
    
    # Create ranges for the pooling window
    kh = tl.arange(0, BLOCK_K)
    kw = tl.arange(0, BLOCK_K)
    
    # Calculate input coordinates for the window
    h_idx = oh * s - p + kh * d
    w_idx = ow * s - p + kw * d
    
    # Create masks for the window:
    # 1. Must be within the specified kernel size (k)
    # 2. Must be within the input tensor boundaries (H, W)
    h_mask = (kh < k) & (h_idx >= 0) & (h_idx < H)
    w_mask = (kw < k) & (w_idx >= 0) & (w_idx < W)
    
    # Combine into a 2D mask for the kernel window
    mask = h_mask[:, None] & w_mask[None, :]
    
    # Calculate flat offsets for the window elements in the input tensor
    # offsets shape: (BLOCK_K, BLOCK_K)
    offsets = offset_base + h_idx[:, None] * W + w_idx[None, :]
    
    # Load window values. Out-of-bounds/padding elements are set to a very small value.
    vals = tl.load(x_ptr + offsets, mask=mask, other=-1e38)
    
    # Perform max reduction over the window
    res = tl.max(vals, axis=0)
    res = tl.max(res, axis=0)
    
    # Store the final max value into the output tensor
    tl.store(out_ptr + out_offset, res)

def triton_maxpool(x, kernel_size, stride, padding, dilation):
    """
    Triton implementation of MaxPool2d.
    """
    x = x.contiguous()
    B, C, H, W = x.shape
    
    # Calculate output dimensions based on PyTorch MaxPool2d formula
    OH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
    
    # BLOCK_K must be a power of 2 for Triton. 
    # We use 32 to cover most common pooling kernel sizes.
    BLOCK_K = 32 
    
    # Grid is defined by (B * C * OH, OW)
    grid = (B * C * OH, OW)
    
    maxpool_kernel[grid](
        x, out,
        B, C, H, W,
        OH, OW,
        kernel_size, stride, padding, dilation,
        BLOCK_K=BLOCK_K
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using custom Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        return triton_maxpool(x, self.kernel_size, self.stride, self.padding, self.dilation)