import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    features,
    input_length,
    output_length,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid dimensions: (batch_size, features, num_blocks)
    pid_b = tl.program_id(0)
    pid_f = tl.program_id(1)
    pid_block = tl.program_id(2)

    # Base offset for the current batch and feature channel
    row_offset = pid_b * features * input_length + pid_f * input_length

    # Output indices for this block
    offsets = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_length

    # Initialize max values to -inf
    max_vals = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)

    # Iterate over kernel elements
    for i in tl.static_range(kernel_size):
        # Compute input indices for the current window position
        # start = j * stride - padding
        # input_idx = row_offset + start + i * dilation
        start = offsets * stride - padding
        input_offsets = row_offset + start + i * dilation

        # Mask for valid input indices
        valid_mask = (input_offsets >= 0) & (input_offsets < input_length)

        # Load values, using -inf for out-of-bounds
        vals = tl.load(x_ptr + input_offsets, mask=valid_mask, other=-float('inf'))

        # Update max
        max_vals = tl.maximum(max_vals, vals)

    # Store results
    tl.store(out_ptr + row_offset + offsets, max_vals, mask=mask)


def triton_maxpool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int
) -> torch.Tensor:
    """
    Wrapper function to launch the Triton Max Pooling 1D kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    batch_size, features, input_length = x.shape
    
    # Calculate output length
    effective_kernel = (kernel_size - 1) * dilation + 1
    output_length = (input_length + 2 * padding - effective_kernel) // stride + 1

    out = torch.empty((batch_size, features, output_length), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128
    num_blocks = (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE

    grid = (batch_size, features, num_blocks)

    maxpool1d_kernel[grid](
        x, out,
        batch_size, features, input_length, output_length,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for Max Pooling 1D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )


def get_inputs():
    batch_size = 64
    features = 192
    sequence_length = 65536
    x = torch.rand(batch_size, features, sequence_length).cuda()
    return [x]


def get_init_inputs():
    kernel_size = 8
    stride = 1
    padding = 4
    dilation = 3
    return_indices = False
    return [kernel_size, stride, padding, dilation, return_indices]