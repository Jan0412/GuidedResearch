import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool_kernel(
    x_ptr, y_ptr,
    N, C, H, W, K, stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1
    num_elements = N * C * H_out * W_out
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Compute output coordinates from linear index
    h_out = (offsets // W_out) % H_out
    w_out = offsets % W_out
    c = (offsets // (H_out * W_out)) % C
    b = offsets // (C * H_out * W_out)
    
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Accumulate sum over the pooling window
    for k in range(K):
        for l in range(K):
            h_in = h_out * stride + k
            w_in = w_out * stride + l
            x_offset = (b * C + c) * H * W + h_in * W + w_in
            val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
            sum_val += val
            
    out = sum_val / (K * K)
    tl.store(y_ptr + offsets, out, mask=mask)

def triton_avg_pool(x: torch.Tensor, kernel_size: int, stride: int = None) -> torch.Tensor:
    if stride is None:
        stride = kernel_size
    assert x.is_cuda
    x = x.contiguous()
    N, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    y = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 256
    num_elements = N * C * H_out * W_out
    grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    avg_pool_kernel[grid](x, y, N, C, H, W, kernel_size, stride, BLOCK_SIZE)
    return y

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool(x, self.kernel_size, self.stride)