import torch
import torch.nn as nn
import math
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    x_ptr, y_ptr,
    B, C, D1, D2, D3,
    Od1, Od2, Od3,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Decompose pid into batch, channel, and spatial output indices
    spatial_size = Od1 * Od2 * Od3
    bc = pid // spatial_size
    b = bc // C
    c = bc % C
    
    idx = pid % spatial_size
    k = idx % Od3
    idx //= Od3
    j = idx % Od2
    i = idx // Od2
    
    # Input start coordinates for the window
    si = i * stride - padding
    sj = j * stride - padding
    sk = k * stride - padding
    
    # Base pointer for current batch and channel
    # Assuming contiguous memory layout: (B, C, D1, D2, D3)
    x_offset = b * (C * D1 * D2 * D3) + c * (D1 * D2 * D3)
    
    # Initialize max value to -inf
    max_val = tl.full((1,), -float('inf'), dtype=tl.float32)
    
    # Iterate over the kernel window
    for di in range(kernel_size):
        for dj in range(kernel_size):
            for dk in range(kernel_size):
                # Compute dilation offsets
                ii = si + di * dilation
                jj = sj + dj * dilation
                kk = sk + dk * dilation
                
                # Create mask for bounds checking
                mask = (ii >= 0) & (ii < D1) & (jj >= 0) & (jj < D2) & (kk >= 0) & (kk < D3)
                
                # Load value with -inf for out-of-bounds
                val = tl.load(x_ptr + x_offset + ii * (D2 * D3) + jj * D3 + kk, mask=mask, other=-float('inf'))
                
                # Update max
                max_val = tl.maximum(max_val, val)
    
    # Store result
    tl.store(y_ptr + x_offset + i * (Od2 * Od3) + j * Od3 + k, max_val)


def triton_maxpool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int) -> torch.Tensor:
    B, C, D1, D2, D3 = x.shape
    
    # Compute output dimensions
    def out_dim(in_dim, k, s, p, d):
        val = (in_dim + 2 * p - d * (k - 1) - 1) / s + 1
        return math.floor(val)
    
    Od1 = out_dim(D1, kernel_size, stride, padding, dilation)
    Od2 = out_dim(D2, kernel_size, stride, padding, dilation)
    Od3 = out_dim(D3, kernel_size, stride, padding, dilation)
    
    assert Od1 > 0 and Od2 > 0 and Od3 > 0, "Output dimensions must be positive"
    
    y = torch.empty((B, C, Od1, Od2, Od3), device=x.device, dtype=x.dtype)
    
    # Ensure contiguous
    x = x.contiguous()
    
    # Grid calculation
    num_blocks = B * C * Od1 * Od2 * Od3
    grid = (num_blocks,)
    
    # Launch kernel
    maxpool3d_kernel[grid](
        x.data_ptr(), y.data_ptr(),
        B, C, D1, D2, D3,
        Od1, Od2, Od3,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=128
    )
    
    return y


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        # return_indices and ceil_mode are not fully implemented in this optimized version
        # as they add complexity. Standard floor-mode without indices is used for performance.
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool3d(x, self.kernel_size, self.stride, self.padding, self.dilation)