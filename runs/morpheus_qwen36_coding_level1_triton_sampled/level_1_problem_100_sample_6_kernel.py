import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    pred_ptr, target_ptr, out_ptr,
    N, M,
    BLOCK_SIZE_M: tl.constexpr,
):
    row_idx = tl.program_id(0)
    acc = 0.0
    
    for start_m in range(0, M, BLOCK_SIZE_M):
        col_idx = start_m + tl.arange(0, BLOCK_SIZE_M)
        mask = col_idx < M
        
        pred = tl.load(pred_ptr + row_idx * M + col_idx, mask=mask, other=0.0)
        target = tl.load(target_ptr + col_idx, mask=mask, other=0.0)
        
        val = 1.0 - pred * target
        val = tl.maximum(val, 0.0)
        acc += tl.sum(val * mask)
        
    tl.store(out_ptr + row_idx, acc)


def triton_hinge_loss(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    N, M = predictions.shape
    out = torch.empty(N, dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE_M = 128
    grid = (N,)
    
    hinge_loss_kernel[grid](
        predictions, targets, out,
        N, M,
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )
    
    return torch.sum(out) / (N * M)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)