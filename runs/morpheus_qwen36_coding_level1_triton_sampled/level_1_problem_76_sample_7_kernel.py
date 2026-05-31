import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    length,
    kernel_size,
    stride,
    dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid dimensions: (batch_size, out_channels, num_tiles)
    b = tl.program_id(0)
    c_out = tl.program_id(1)
    tile_idx = tl.program_id(2)

    # Output tile offsets
    out_offsets = tile_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < ((length - (kernel_size - 1) * dilation - 1) // stride + 1)

    # Input offsets calculation: [out_idx * stride + k * dilation]
    # Shape: (BLOCK_SIZE, kernel_size)
    in_offsets = out_offsets[:, None] * stride + tl.arange(0, kernel_size)[None, :] * dilation
    in_mask = in_offsets < length

    # Load input tile
    x_tile = tl.load(x_ptr + in_offsets, mask=in_mask, other=0.0)

    # Weight offsets calculation: [c_out * in_channels * kernel_size + c_in * kernel_size + k]
    # Shape: (in_channels, kernel_size)
    w_offsets = (
        c_out * in_channels * kernel_size
        + tl.arange(0, in_channels)[:, None] * kernel_size
        + tl.arange(0, kernel_size)[None, :]
    )
    w_tile = tl.load(w_ptr + w_offsets)

    # Compute convolution: sum over in_channels and kernel_size
    # x_tile: (BLOCK_SIZE, kernel_size)
    # w_tile: (in_channels, kernel_size)
    # Result: (BLOCK_SIZE,)
    acc = tl.sum(x_tile[:, None, :] * w_tile[None, :, :], axis=2)
    acc = tl.sum(acc, axis=1)

    # Add bias
    out = acc + tl.load(b_ptr + c_out)

    # Store output
    tl.store(out_ptr + out_offsets, out, mask=mask)


def triton_conv1d(x, w, b, stride, dilation, kernel_size):
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()

    batch_size, in_channels, length = x.shape
    out_channels, _, _ = w.shape
    out_length = (length - (kernel_size - 1) * dilation - 1) // stride + 1

    out = torch.empty(batch_size, out_channels, out_length, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128
    num_tiles = (out_length + BLOCK_SIZE - 1) // BLOCK_SIZE

    grid = (batch_size, out_channels, num_tiles)

    conv1d_kernel[grid](
        x, w, b, out,
        batch_size, in_channels, out_channels, length,
        kernel_size, stride, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.bias_param = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d(x, self.weight, self.bias_param, self.stride, self.dilation, self.kernel_size)


def get_inputs():
    batch_size = 64
    in_channels = 64
    length = 524280
    x = torch.rand(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    stride = 3
    dilation = 4
    bias = False
    return [in_channels, out_channels, kernel_size, stride, dilation, bias]