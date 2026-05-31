import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def kl_div_kernel(
    pred_ptr, target_ptr, out_ptr,
    batch_size, seq_len,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * seq_len
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + seq_len
    
    # Load inputs with safe fallback for masked elements
    preds = tl.load(pred_ptr + offsets, mask=mask, other=1e-12)
    targets = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Clamp to avoid log(0) and ensure numerical stability
    preds = tl.maximum(preds, 1e-12)
    targets = tl.maximum(targets, 1e-12)
    
    # Compute log probabilities
    log_preds = tl.log(preds)
    log_targets = tl.log(targets)
    
    # KL divergence term: target * (log(target) - log(pred))
    terms = targets * (log_targets - log_preds)
    
    # Zero out contributions from masked-out elements
    terms = tl.where(mask, terms, 0.0)
    
    # Reduce along the sequence dimension
    row_sum = tl.sum(terms, axis=0)
    
    # Store per-row sums
    tl.store(out_ptr + pid, row_sum)


def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size = predictions.shape[0]
    seq_len = predictions.shape[1]
    
    # Output tensor to hold per-row KL sums
    out = torch.empty(batch_size, dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    # Launch Triton kernel
    kl_div_kernel[grid](
        predictions, targets, out,
        batch_size, seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum across batch dimension and divide by batch size (batchmean reduction)
    return out.sum() / batch_size


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)