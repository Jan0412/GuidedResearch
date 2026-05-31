import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def kl_div_kernel(
    predictions_ptr, targets_ptr, out_ptr,
    batch_size, input_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid < batch_size:
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < input_size
        
        preds = tl.load(predictions_ptr + pid * input_size + offsets, mask=mask, other=0.0)
        tgts = tl.load(targets_ptr + pid * input_size + offsets, mask=mask, other=0.0)
        
        log_preds = tl.log(preds)
        log_tgts = tl.log(tgts)
        
        row_kl = tgts * (log_tgts - log_preds)
        row_sum = tl.sum(row_kl)
        
        tl.store(out_ptr + pid, row_sum)


def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous().float()
    targets = targets.contiguous().float()
    
    batch_size = predictions.shape[0]
    input_size = predictions.shape[1]
    
    out = torch.empty(batch_size, dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = 16384
    
    grid = (batch_size,)
    
    kl_div_kernel[grid](predictions, targets, out, batch_size, input_size, BLOCK_SIZE=BLOCK_SIZE)
    
    return torch.sum(out) / batch_size


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)