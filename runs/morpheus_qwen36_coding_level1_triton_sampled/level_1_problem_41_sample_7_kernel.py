import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr, out_ptr,
    B, F, L_in, L_out,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    tile_idx = tl.program_id(1)
    
    b = pid // F
    f = pid % F
    
    # Base pointer for the current batch and feature channel
    x_ptr += b * F * L_in + f * L_in
    
    # Output offsets for the current tile
    out_offsets = tile_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < L_out
    
    # Initialize output values to negative infinity
    out_vals = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    
    # Slide over the kernel
    for k in range(kernel_size):
        # Compute input indices for the current kernel position
        input_offsets = out_offsets * stride - padding + k * dilation
        # Mask for valid input indices
        input_mask = (input_offsets >= 0) & (input_offsets < L_in)
        # Load values, using -inf for out-of-bounds to preserve max operation
        vals = tl.load(x_ptr + input_offsets, mask=input_mask, other=-float('inf'))
        # Accumulate maximum
        out_vals = tl.maximum(out_vals, vals)
        
    # Store the result
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


def triton_maxpool1d(x, kernel_size, stride, padding, dilation):
    assert x.is_cuda and x.is_contiguous(), "Input tensor must be contiguous and on CUDA."
    B, F, L_in = x.shape
    
    # Calculate output sequence length matching PyTorch's formula
    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    out = torch.empty((B, F, L_out), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    num_bf = B * F
    num_tiles = (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (num_bf, num_tiles)
    
    maxpool1d_kernel[grid](x, out, B, F, L_in, L_out, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)