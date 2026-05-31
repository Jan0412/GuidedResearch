import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    stride_x,
    stride_mask,
    stride_out,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for masked cumulative sum along the last dimension.
    Each program handles one batch element.
    """
    pid = tl.program_id(0)
    x_ptr += pid * stride_x
    mask_ptr += pid * stride_mask
    out_ptr += pid * stride_out

    global_acc = 0.0

    for start in range(0, dim_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size

        # Load x and mask
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        m_vals = tl.load(mask_ptr + offsets, mask=mask, other=0.0)

        # Compute element-wise product
        prod = x_vals * m_vals

        # Compute local prefix sum
        local_acc = 0.0
        # Use a local array to store intermediate results
        local_out = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for i in range(BLOCK_SIZE):
            if start + i < dim_size:
                local_acc += prod[i]
                local_out[i] = local_acc + global_acc

        # Store results
        tl.store(out_ptr + offsets, local_out, mask=mask)

        # Update global accumulator
        global_acc = local_acc + global_acc


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Optimized masked cumulative sum using Triton kernel.
    Assumes dim=1 (last dimension) for optimal performance.
    """
    assert dim == 1, "Optimized kernel currently supports only dim=1."
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    assert x.shape == mask.shape, "x and mask must have the same shape."

    x = x.contiguous()
    mask = mask.contiguous()

    batch_size = x.shape[0]
    dim_size = x.shape[1]

    out = torch.empty_like(x)

    # Choose block size based on dim_size for efficiency
    BLOCK_SIZE = min(1024, triton.next_power_of_2(dim_size))

    grid = (batch_size,)

    masked_cumsum_kernel[grid](
        x, mask, out,
        x.stride(0), mask.stride(0), out.stride(0),
        dim_size, BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return triton_masked_cumsum(x, mask, self.dim)