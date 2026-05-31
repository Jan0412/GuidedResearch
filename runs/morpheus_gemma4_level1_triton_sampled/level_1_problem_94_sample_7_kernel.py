import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_kernel(
    pred_ptr, 
    target_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    # Calculate the offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask for out-of-bounds elements
    mask = offsets < n_elements

    # Load data
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # Compute squared difference: (pred - target)^2
    diff = pred - target
    sq_diff = diff * diff

    # Sum the squared differences within the block
    # We use tl.sum to reduce the block to a single scalar
    block_sum = tl.sum(sq_diff, axis=0)

    # Atomically add the block sum to the global accumulator
    # Since out_ptr is a pointer to a single scalar, we use atomic_add
    tl.atomic_add(out_ptr, block_sum)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Triton implementation of Mean Squared Error.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous and FP32
    predictions = predictions.contiguous().float()
    targets = targets.contiguous().float()

    n_elements = predictions.numel()
    
    # Output tensor to hold the global sum
    # Initialized to 0.0
    out = torch.zeros((), device=predictions.device, dtype=torch.float32)

    # Tunable block size
    BLOCK_SIZE = 1024
    
    # Grid configuration: one block per group of BLOCK_SIZE elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    mse_kernel[grid](
        predictions, 
        targets, 
        out, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    # The kernel computes the sum of squared errors; we divide by n_elements for the mean
    return out / n_elements

class ModelNew(nn.Module):
    """
    An optimized model that computes the Mean Squared Error loss using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)