import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr, y_ptr,
    B, C, L_in, L_out,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr
):
    pid_bc = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    
    if pid_bc >= B * C:
        return
        
    x_ptr += pid_bc * L_in
    y_ptr += pid_bc * L_out + pid_chunk * BLOCK_SIZE
    
    offsets = pid_chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < L_out
    
    max_vals = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    
    for k in range(kernel_size):
        orig_idx = offsets - padding + k * dilation
        val = tl.load(x_ptr + orig_idx, mask=(orig_idx >= 0) & (orig_idx < L_in), other=0.0)
        max_vals = tl.maximum(max_vals, val)
        
    tl.store(y_ptr + offsets, max_vals, mask=mask)


def triton_maxpool1d(x, kernel_size, stride, padding, dilation):
    assert x.is_cuda
    x = x.contiguous()
    B, C, L_in = x.shape
    eff_k = kernel_size + (kernel_size - 1) * (dilation - 1)
    L_out = (L_in + 2 * padding - eff_k) // stride + 1
    y = torch.empty((B, C, L_out), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    num_chunks = (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (B * C, num_chunks)
    
    maxpool1d_kernel[grid](
        x, y,
        B, C, L_in, L_out,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)