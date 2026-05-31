import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(
    pred_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute 1 - pred * target
    product = pred * target
    diff = 1.0 - product
    
    # Clamp to minimum 0
    clamped = tl.maximum(diff, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, clamped, mask=mask)

@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Use atomic operations to accumulate sum
    # Since we're computing mean, we'll do it in two steps
    # First, compute sum using atomics
    tl.atomic_add(output_ptr, tl.sum(x, axis=0))
    
    # For mean computation, we'll handle it in post-processing

def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Prepare output tensor for intermediate results
    out = torch.empty(predictions.shape[0], dtype=torch.float32, device=predictions.device)
    
    # Number of elements in the tensor
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel for hinge loss computation
    hinge_loss_kernel[grid](predictions, targets, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute mean manually since Triton doesn't support reduction efficiently in this case
    return torch.mean(out)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)