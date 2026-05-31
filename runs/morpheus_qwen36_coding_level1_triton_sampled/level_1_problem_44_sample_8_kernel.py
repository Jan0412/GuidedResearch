import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr, out_ptr,
    batch_size, in_channels, input_length, output_length,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= batch_size * in_channels:
        return
        
    b = pid // in_channels
    c = pid % in_channels
    
    x_base = x_ptr + b * in_channels * input_length + c * input_length
    out_base = out_ptr + b * in_channels * output_length + c * output_length
    
    for start_i in range(0, output_length, BLOCK_SIZE):
        block_offsets = start_i + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < output_length
        
        win_start = start_i * stride + padding
        win_len = BLOCK_SIZE * stride + kernel_size
        win_mask = tl.arange(0, win_len) < input_length
        
        win_ptrs = x_base + win_start + tl.arange(0, win_len)
        win_data = tl.load(win_ptrs, mask=win_mask, other=0.0)
        
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for k in range(kernel_size):
            acc += win_data[block_offsets + k]
            
        out = acc / kernel_size
        tl.store(out_base + block_offsets, out, mask=mask)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty((batch_size, in_channels, output_length), dtype=x.dtype, device=x.device)
    
    grid = lambda meta: (batch_size * in_channels,)
    
    avg_pool1d_kernel[grid](
        x, out,
        batch_size, in_channels, input_length, output_length,
        kernel_size, stride, padding,
        BLOCK_SIZE=128
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)