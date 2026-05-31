import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def smooth_l1_kernel(
    x_ptr,
    y_ptr,
    partial_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate offsets for the current block
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Compute absolute difference
    diff = tl.abs(x - y)

    # Smooth L1 loss calculation:
    # loss = 0.5 * diff^2 if diff < 1.0 else diff - 0.5
    loss = tl.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5)

    # Sum the loss values in the current block
    block_sum = tl.sum(loss)

    # Store the block sum in the partial sums array
    tl.store(partial_sums_ptr + tl.program_id(0), block_sum)


def triton_smooth_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes Smooth L1 Loss using a custom Triton kernel.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Allocate memory for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device='cuda')

    # Define grid
    grid = (num_blocks,)

    # Launch kernel
    smooth_l1_kernel[grid](x, y, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Reduce partial sums to get the total sum and compute mean
    total_sum = partial_sums.sum()
    return total_sum / n_elements


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1(predictions, targets)