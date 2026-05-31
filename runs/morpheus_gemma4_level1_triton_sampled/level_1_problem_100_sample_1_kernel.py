import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(
    pred_ptr, 
    target_ptr, 
    out_ptr,
    M, 
    N,
    grid_n,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for boundary conditions
    mask_m = rm < M
    mask_n = rn < N

    # Load predictions tile: (BLOCK_M, BLOCK_N)
    # Offset calculation for 2D tensor predictions: row * N + col
    pred_offsets = rm[:, None] * N + rn[None, :]
    p = tl.load(pred_ptr + pred_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # Load targets tile: (BLOCK_N,)
    t = tl.load(target_ptr + rn, mask=mask_n, other=0.0)

    # Broadcast targets to (1, BLOCK_N) for multiplication
    t = t[None, :]

    # Compute Hinge Loss: max(0, 1 - pred * target)
    val = 1.0 - p * t
    val = tl.maximum(0.0, val)

    # Mask out values outside the actual tensor dimensions to avoid affecting the sum
    val = tl.where(mask_m[:, None] & mask_n[None, :], val, 0.0)

    # Partial reduction within the block
    # Sum across rows (M), then across columns (N)
    block_sum = tl.sum(val, axis=0)
    block_sum = tl.sum(block_sum, axis=0)

    # Store the scalar partial sum for this block in the output tensor
    # The output tensor is viewed as a 1D array of size grid_m * grid_n
    tl.store(out_ptr + pid_m * grid_n + pid_n, block_sum)

def triton_hinge_loss_mean(predictions: torch.Tensor, targets: torch.Tensor):
    # Ensure tensors are contiguous on GPU
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    M, N = predictions.shape
    
    # Tunable block sizes
    BLOCK_M = 128
    BLOCK_N = 128

    # Calculate grid dimensions
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m, grid_n)

    # Allocate space for partial sums from each block
    partial_sums = torch.empty((grid_m * grid_n,), device=predictions.device, dtype=torch.float32)

    # Launch the Triton kernel
    hinge_loss_kernel[grid](
        predictions, 
        targets, 
        partial_sums, 
        M, 
        N, 
        grid_n, 
        BLOCK_M=BLOCK_M, 
        BLOCK_N=BLOCK_N
    )

    # Final reduction: sum all partial sums and divide by total elements
    total_sum = torch.sum(partial_sums)
    return total_sum / (M * N)

class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace PyTorch operators with the optimized Triton implementation
        return triton_hinge_loss_mean(predictions, targets)