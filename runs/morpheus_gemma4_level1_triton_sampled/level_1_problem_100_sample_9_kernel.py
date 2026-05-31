import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(
    predictions_ptr, 
    targets_ptr, 
    out_ptr, 
    N, M, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Fused kernel to compute Hinge Loss: mean(max(0, 1 - predictions * targets)).
    """
    # Program IDs for the 2D grid
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Calculate offsets for the current block
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Boundary masks to handle tensors not perfectly divisible by block size
    mask_n = offs_n < N
    mask_m = offs_m < M
    
    # Load targets: shape (BLOCK_SIZE_M,)
    # targets is broadcasted across the N dimension
    targets = tl.load(targets_ptr + offs_m, mask=mask_m, other=0.0)
    
    # Load predictions: shape (BLOCK_SIZE_N, BLOCK_SIZE_M)
    # Using 2D indexing to fetch a block of data from the predictions tensor
    preds = tl.load(predictions_ptr + offs_n[:, None] * M + offs_m[None, :], 
                    mask=mask_n[:, None] & mask_m[None, :], other=0.0)
    
    # Element-wise Hinge Loss calculation: max(0, 1 - pred * target)
    # targets[None, :] expands (BLOCK_SIZE_M,) to (1, BLOCK_SIZE_M) for broadcasting
    val = 1.0 - preds * targets[None, :]
    val = tl.maximum(0.0, val)
    
    # Local reduction within the block to minimize atomic operations
    # Sum across the N dimension, then across the M dimension
    block_sum = tl.sum(val, axis=0)
    block_sum = tl.sum(block_sum, axis=0)
    
    # Atomically add the local block sum to the global result scalar
    tl.atomic_add(out_ptr, block_sum)

class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss using a fused Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Ensure tensors are contiguous on the GPU for optimal memory access patterns
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        
        N, M = predictions.shape
        # Initialize a scalar tensor to accumulate the sum of losses
        out = torch.zeros((), device=predictions.device, dtype=torch.float32)
        
        # Block sizes tuned for typical GPU architectures to maximize throughput
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_M = 128
        
        # Define the grid of blocks to cover the entire predictions tensor
        grid = ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N, 
                (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
        
        # Launch the fused Triton kernel
        hinge_loss_kernel[grid](
            predictions, targets, out, 
            N, M, 
            BLOCK_SIZE_N=BLOCK_SIZE_N, 
            BLOCK_SIZE_M=BLOCK_SIZE_M
        )
        
        # The kernel computes the total sum; we return the mean over all elements
        return out / (N * M)