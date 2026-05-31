import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr, out_ptr,
    batch_size, in_channels, input_length, output_length,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
    LOAD_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_out_blocks = tl.cdiv(output_length, BLOCK_SIZE)
    batch = pid // (in_channels * num_out_blocks)
    channel = (pid // num_out_blocks) % in_channels
    out_block_idx = pid % num_out_blocks
    
    out_start = out_block_idx * BLOCK_SIZE
    in_start = out_start * stride - padding
    
    # Offsets for output
    out_offsets = out_start + tl.arange(0, BLOCK_SIZE)
    mask_out = out_offsets < output_length
    
    # Offsets for input buffer
    in_offsets = in_start + tl.arange(0, LOAD_SIZE)
    mask_in = (in_offsets >= 0) & (in_offsets < input_length)
    
    # Base pointer for the specific batch and channel
    base_ptr = x_ptr + (batch * in_channels + channel) * input_length
    
    # Load input buffer with masking for padding and boundaries
    x_buf = tl.load(base_ptr + in_offsets, mask=mask_in, other=0.0)
    
    # Compute sum over kernel window
    res = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(kernel_size):
        buf_idx = out_offsets * stride + k
        # buf_idx is always within [0, LOAD_SIZE) by construction
        res += x_buf[buf_idx]
        
    # Compute average
    out = res / kernel_size
    
    # Store result
    tl.store(out_ptr + out_start + tl.arange(0, BLOCK_SIZE), out, mask=mask_out)


def triton_avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    batch_size, in_channels, input_length = x.shape
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty(batch_size, in_channels, output_length, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    raw_load_size = kernel_size + (BLOCK_SIZE - 1) * stride
    # Round up to next power of 2 for better memory alignment
    LOAD_SIZE = 1 << (raw_load_size - 1).bit_length()
    
    num_out_blocks = (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_size * in_channels * num_out_blocks,)
    
    avg_pool1d_kernel[grid](
        x, out,
        batch_size, in_channels, input_length, output_length,
        kernel_size, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE,
        LOAD_SIZE=LOAD_SIZE
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