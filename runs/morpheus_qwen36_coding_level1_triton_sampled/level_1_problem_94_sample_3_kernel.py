import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_kernel(
    preds_ptr,
    targets_ptr,
    partial_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    preds = tl.load(preds_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    diff = preds - targets
    sq_diff = diff * diff
    
    acc = tl.sum(sq_diff)
    tl.store(partial_sums_ptr + pid, acc)


def triton_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert preds.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    preds = preds.contiguous()
    targets = targets.contiguous()
    
    n_elements = preds.numel()
    BLOCK_SIZE = 1 << 14  # 16384
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device='cuda')
    
    grid = lambda meta: (num_blocks,)
    
    mse_kernel[grid](
        preds, targets, partial_sums, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    
    return partial_sums.sum() / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)


def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape) * scale, torch.rand(batch_size, *input_shape)]


def get_init_inputs():
    return []