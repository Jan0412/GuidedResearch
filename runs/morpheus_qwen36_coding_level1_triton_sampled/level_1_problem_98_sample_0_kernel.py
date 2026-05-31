import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def kl_div_kernel(
    pred_ptr, target_ptr, partial_sums_ptr,
    batch_size, input_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * input_size
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute KL divergence terms: target * (log(pred) - log(target))
    terms = target * (tl.log(pred) - tl.log(target))
    
    # Reduce within the block
    partial_sum = tl.sum(terms, axis=0)
    
    tl.store(partial_sums_ptr + pid, partial_sum)


def triton_kl_div(predictions, targets):
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size = predictions.shape[0]
    input_size = predictions.shape[1]
    total_elements = batch_size * input_size
    
    BLOCK_SIZE = 4096
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device='cuda')
    
    grid = (num_blocks,)
    kl_div_kernel[grid](
        predictions, targets, partial_sums,
        batch_size, input_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return partial_sums.sum().item() / batch_size


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)