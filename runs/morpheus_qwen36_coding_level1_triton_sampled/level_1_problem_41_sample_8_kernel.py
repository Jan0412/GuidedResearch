import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,
    out_ptr,
    stride_batch,
    stride_channel,
    stride_seq,
    L,
    L_out,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    DILATION: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_l = (L_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    b_c = pid // num_blocks_l
    l_block = pid % num_blocks_l

    b = b_c // stride_channel
    c = b_c % stride_channel

    l_out_start = l_block * BLOCK_SIZE

    offsets = l_out_start + tl.arange(0, BLOCK_SIZE)
    mask_out = offsets < L_out

    input_start = l_out_start * STRIDE - PADDING
    window_size = BLOCK_SIZE + (KERNEL_SIZE - 1) * DILATION
    input_offsets = input_start + tl.arange(0, window_size)
    mask_input = input_offsets < L

    x_ptr_b_c = x_ptr + b * stride_batch + c * stride_channel
    window = tl.load(x_ptr_b_c + input_offsets, mask=mask_input, other=-float('inf'))

    max_vals = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)

    for j in range(KERNEL_SIZE):
        idx = j * DILATION
        vals = window[tl.arange(0, BLOCK_SIZE) + idx]
        max_vals = tl.maximum(max_vals, vals)

    tl.store(out_ptr + offsets, max_vals, mask=mask_out)


def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    batch_size, features, sequence_length = x.shape
    
    L_out = (sequence_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out = torch.empty((batch_size, features, L_out), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128
    num_blocks = (batch_size * features * L_out + BLOCK_SIZE - 1) // BLOCK_SIZE

    grid = (num_blocks,)

    maxpool1d_kernel[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2),
        sequence_length, L_out,
        BLOCK_SIZE=BLOCK_SIZE,
        KERNEL_SIZE=kernel_size,
        STRIDE=stride,
        PADDING=padding,
        DILATION=dilation,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)


batch_size = 64
features = 192
sequence_length = 65536

kernel_size = 8
stride = 1
padding = 4
dilation = 3
return_indices = False


def get_inputs():
    x = torch.rand(batch_size, features, sequence_length)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]