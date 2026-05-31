import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardtanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # HardTanh activation: clamp values between -1.0 and 1.0
    out = tl.minimum(1.0, tl.maximum(-1.0, x))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_hardtanh(x: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton implementation of HardTanh activation.
    Optimized for FP32 precision and large tensor shapes.
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size for memory throughput
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hardtanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardtanh(x)