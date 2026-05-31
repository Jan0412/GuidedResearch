import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    pred_ptr, target_ptr, out_ptr, n_elements, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for start in range(pid * BLOCK_SIZE, n_elements, num_blocks * BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
        
        diff = tl.abs(pred - target)
        cond = diff < beta
        loss = tl.where(cond, 0.5 * diff * diff, diff - 0.5 * beta)
        
        thread_sum = tl.sum(loss, axis=0)
        tl.atomic_add(out_ptr, thread_sum)


def triton_smooth_l1_loss(predictions, targets, beta=1.0):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    out = torch.zeros((), device=predictions.device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)
    
    smooth_l1_loss_kernel[grid](predictions, targets, out, n_elements, beta=beta, BLOCK_SIZE=BLOCK_SIZE)
    
    return out / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)


batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []