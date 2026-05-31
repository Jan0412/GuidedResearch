import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(
    p_ptr,  # Pointer to predictions (N, D)
    t_ptr,  # Pointer to targets (D,)
    out_ptr,  # Pointer to scalar sum
    N,  # Number of rows
    D,  # Number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for the current block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Mask for boundary conditions
    mask_m = offs_m < N
    mask_n = offs_n < D

    # Compute 2D indices for predictions tensor
    # p is (N, D), so index is row * D + col
    p_offsets = offs_m[:, None] * D + offs_n[None, :]
    
    # Load prediction values
    p = tl.load(p_ptr + p_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # Load target values (broadcasting across rows)
    t = tl.load(t_ptr + offs_n, mask=mask_n, other=0.0)

    # Compute: max(0, 1 - predictions * targets)
    # t is shape (BLOCK_SIZE_N,), p is (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # Broadcasting t to (1, BLOCK_SIZE_N)
    val = 1.0 - p * t[None, :]
    val = tl.maximum(0.0, val)

    # Apply mask to avoid counting out-of-bounds elements in the sum
    val = tl.where(mask_m[:, None] & mask_n[None, :], val, 0.0)

    # Reduce the block sum
    block_sum = tl.sum(val, axis=0)
    block_sum = tl.sum(block_sum, axis=0)

    # Atomically add the block sum to the global total
    tl.atomic_add(out_ptr, block_sum)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Triton wrapper for the Hinge Loss calculation.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    N, D = predictions.shape
    
    # Output tensor to hold the accumulated sum
    out = torch.zeros(1, device=predictions.device, dtype=torch.float32)

    # Block sizes for the grid
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64

    # Define the grid: (number of blocks in M, number of blocks in N)
    grid = (
        (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (D + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )

    # Launch the kernel
    hinge_loss_kernel[grid](
        predictions, 
        targets, 
        out, 
        N, 
        D, 
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )

    # The result is the mean: sum / total_elements
    return out.item() / (N * D)


class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace torch.mean(torch.clamp(1 - predictions * targets, min=0)) 
        # with the fused Triton implementation.
        return torch.tensor(triton_hinge_loss(predictions, targets), device=predictions.device)