import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr, y_ptr,
    batch_size, in_channels,
    input_length, output_length,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    num_blocks_per_channel = triton.cdiv(output_length, BLOCK_SIZE)
    
    block_idx = pid % num_blocks_per_channel
    channel_idx = (pid // num_blocks_per_channel) % in_channels
    batch_idx = pid // (in_channels * num_blocks_per_channel)
    
    start_out_idx = block_idx * BLOCK_SIZE
    
    x_base = x_ptr + batch_idx * in_channels * input_length + channel_idx * input_length
    y_base = y_ptr + batch_idx * in_channels * output_length + channel_idx * output_length
    
    # Compute sum for first output in block
    current_sum = 0.0
    for k in range(kernel_size):
        in_idx = start_out_idx * stride + k - padding
        mask = (in_idx >= 0) & (in_idx < input_length)
        val = tl.load(x_base + in_idx, mask=mask, other=0.0)
        current_sum += val
        
    # Store first
    out_idx = start_out_idx
    if out_idx < output_length:
        tl.store(y_base + out_idx, current_sum / kernel_size)
        
    # Slide window for remaining outputs in block
    for i in range(1, BLOCK_SIZE):
        out_idx = start_out_idx + i
        if out_idx >= output_length:
            break
            
        leave_idx = (out_idx - 1) * stride - padding
        enter_idx = out_idx * stride + kernel_size - 1 - padding
        
        mask_leave = (leave_idx >= 0) & (leave_idx < input_length)
        leave_val = tl.load(x_base + leave_idx, mask=mask_leave, other=0.0)
        
        mask_enter = (enter_idx >= 0) & (enter_idx < input_length)
        enter_val = tl.load(x_base + enter_idx, mask=mask_enter, other=0.0)
        
        current_sum = current_sum - leave_val + enter_val
        tl.store(y_base + out_idx, current_sum / kernel_size)


def triton_avg_pool1d(x, kernel_size, stride, padding):
    batch_size, in_channels, input_length = x.shape
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    y = torch.empty((batch_size, in_channels, output_length), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    num_blocks_per_channel = triton.cdiv(output_length, BLOCK_SIZE)
    num_blocks = batch_size * in_channels * num_blocks_per_channel
    
    grid = (num_blocks,)
    
    avg_pool1d_kernel[grid](
        x, y,
        batch_size, in_channels,
        input_length, output_length,
        kernel_size, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool1d(x, self.kernel_size, self.stride, self.padding)