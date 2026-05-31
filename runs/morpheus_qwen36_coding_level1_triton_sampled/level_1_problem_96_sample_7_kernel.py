import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def smooth_l1_loss_kernel(
    pred_ptr, target_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    diff = tl.abs(pred - target)
    loss = tl.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5)
    
    block_sum = tl.sum(loss)
    tl.store(out_ptr + pid, block_sum)

def triton_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.is_cuda and target.is_cuda, "Tensors must be on CUDA."
    pred = pred.contiguous()
    target = target.contiguous()
    
    n_elements = pred.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    block_sums = torch.zeros(num_blocks, device=pred.device, dtype=torch.float32)
    
    grid = lambda meta: (num_blocks,)
    smooth_l1_loss_kernel[grid](pred, target, block_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    total_sum = block_sums.sum().item()
    return torch.tensor(total_sum / n_elements, device=pred.device, dtype=torch.float32)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)