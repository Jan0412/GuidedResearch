import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    Ho, Wo,
    kernel_size: tl.constexpr, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_outputs = N * C * Ho * Wo
    
    for block_idx in tl.range(0, BLOCK_SIZE, 1):
        curr_pid = pid * BLOCK_SIZE + block_idx
        if curr_pid >= num_outputs:
            break
            
        rem = curr_pid
        wo = rem % Wo
        rem = rem // Wo
        ho = rem % Ho
        rem = rem // Ho
        c = rem % C
        n = rem // C
        
        base_ptr = x_ptr + (n * C + c) * H * W
        
        max_val = 0.0
        
        for i in tl.range(kernel_size):
            for j in tl.range(kernel_size):
                h_idx = ho * stride - padding + i * dilation
                w_idx = wo * stride - padding + j * dilation
                
                h_valid = (h_idx >= 0) & (h_idx < H)
                w_valid = (w_idx >= 0) & (w_idx < W)
                valid = h_valid & w_valid
                
                val = tl.load(base_ptr + h_idx * W + w_idx, mask=valid, other=0.0)
                max_val = tl.maximum(max_val, val)
                
        y_offset = (n * C + c) * Ho * Wo + ho * Wo + wo
        tl.store(y_ptr + y_offset, max_val)


def triton_maxpool2d(x: torch.Tensor, kernel_size, stride, padding, dilation):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    
    N, C, H, W = x.shape
    Ho = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    Wo = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    y = torch.empty((N, C, Ho, Wo), dtype=x.dtype, device=x.device)
    
    num_outputs = N * C * Ho * Wo
    BLOCK_SIZE = 128
    
    grid = (num_outputs + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    maxpool2d_kernel[grid](x, y, N, C, H, W, Ho, Wo, kernel_size, stride, padding, dilation, BLOCK_SIZE)
    
    return y


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)