import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_kernel(
    x_ptr,
    out_ptr,
    B,
    D2,
    D1,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d2 = tl.program_id(1)

    # Base offset for the slice x[pid_b, pid_d2, :] in the transposed tensor
    # Transposed shape is (B, D2, D1), contiguous strides are (D2*D1, D1, 1)
    base_offset = pid_b * D2 * D1 + pid_d2 * D1

    # Initialize min accumulator to infinity
    min_val = tl.full((1,), float('inf'), dtype=tl.float32)

    # Loop over the reduction dimension D1 in blocks
    for start in range(0, D1, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D1

        # Load chunk of data; use inf for out-of-mask elements
        chunk = tl.load(x_ptr + base_offset + offsets, mask=mask, other=float('inf'))

        # Update min accumulator: broadcast min_val to chunk size, then reduce
        min_val = tl.reduce(tl.minimum(min_val, chunk), axis=0, combine_fn=tl.minimum)

    # Store the final min value to the output tensor
    # Output shape is (B, D2), strides are (D2, 1)
    tl.store(out_ptr + pid_b * D2 + pid_d2, min_val)


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for min reduction over dim=1.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            # Transpose to make the reduction dimension contiguous for efficient memory access
            # Input shape: (B, D1, D2) -> Transposed: (B, D2, D1)
            x_t = x.transpose(1, 2)
            B, D2, D1 = x_t.shape

            # Prepare output tensor
            out = torch.empty((B, D2), dtype=x.dtype, device=x.device)

            # Tunable block size for the reduction dimension
            BLOCK_SIZE = 128

            # Grid: one block per output element (B, D2)
            grid = (B, D2)

            # Launch the Triton kernel
            min_kernel[grid](x_t, out, B, D2, D1, BLOCK_SIZE)

            return out
        else:
            # Fallback to PyTorch for other dimensions
            return torch.min(x, dim=self.dim)[0]