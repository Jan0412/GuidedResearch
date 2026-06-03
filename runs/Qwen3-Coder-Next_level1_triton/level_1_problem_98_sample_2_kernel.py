import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def kl_divergence_kernel(
    predictions_ptr,  # Pointer to predictions (P)
    targets_ptr,      # Pointer to targets (Q)
    out_ptr,          # Pointer to output
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global index for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    p = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    q = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute KL divergence: p * (log(p) - log(q))
    # Use log_softmax style computation for numerical stability
    # KL = p * log(p/q) = p * (log(p) - log(q))
    log_p = tl.log(p)
    log_q = tl.log(q)
    kl = p * (log_p - log_q)
    
    # Store intermediate result (will be reduced later)
    tl.store(out_ptr + offsets, kl, mask=mask)


@triton.jit
def sum_reduce_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    stride: tl.constexpr,
):
    # Each block computes partial sum for one sample in batch
    sample_id = tl.program_id(0)
    
    # Calculate starting offset for this sample
    start_idx = sample_id * stride
    
    # Accumulate sum for this sample
    sum_val = tl.zeros([1], dtype=tl.float32)
    
    # Process in chunks
    for i in range(0, BLOCK_SIZE, 128):
        offsets = start_idx + i + tl.arange(0, 128)
        mask = offsets < (start_idx + BLOCK_SIZE)
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0, keepdims=False)
    
    # Store result
    tl.store(out_ptr + sample_id, sum_val)


def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute KL divergence using Triton kernels with batchmean reduction.
    
    KL(P||Q) = (1/batch_size) * Σ_i Σ_j P[i,j] * (log(P[i,j]) - log(Q[i,j]))
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Input shapes must match."
    
    # Ensure contiguous memory
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Get dimensions
    batch_size = predictions.shape[0]
    n_elements = predictions.numel()
    
    # First kernel: compute element-wise KL divergence
    # Output has same shape as input
    kl_intermediate = torch.empty_like(predictions)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    kl_divergence_kernel[grid](
        predictions, targets, kl_intermediate, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second kernel: reduce along last dimension for each batch element
    # Since input_shape might not be 1D, we need to handle multi-dimensional cases
    # For simplicity, we'll assume the last dimension is the feature dimension
    # and reduce over it, then average over batch
    
    # Calculate total elements per batch sample
    elements_per_sample = n_elements // batch_size
    
    # Create output for per-batch sums
    batch_sums = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)
    
    # Adjust grid for batch-wise reduction
    # For each batch sample, we need to reduce over its elements
    grid_batch = (batch_size,)
    
    # Use a reasonable block size for reduction
    BLOCK_REDUCE = 256
    
    # Custom reduction kernel for each batch sample
    @triton.jit
    def batch_reduction_kernel(
        in_ptr,
        out_ptr,
        batch_idx,
        elements_per_batch,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each program handles one batch sample
        # We're already indexed by batch sample via program_id
        
        # Accumulate sum for this batch sample
        sum_val = tl.zeros([1], dtype=tl.float32)
        
        # Process in chunks
        for i in range(0, elements_per_batch, 128):
            offsets = batch_idx * elements_per_batch + i + tl.arange(0, 128)
            mask = offsets < (batch_idx + 1) * elements_per_batch
            x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
            sum_val += tl.sum(x, axis=0)
        
        tl.store(out_ptr + batch_idx, sum_val)
    
    # Launch batch reduction kernel
    batch_reduction_kernel[grid_batch](
        kl_intermediate, batch_sums, 
        tl.arange(0, 1),  # Dummy parameter - we'll use program_id instead
        elements_per_sample,
        BLOCK_SIZE=BLOCK_REDUCE
    )
    
    # Actually, let's rewrite this more cleanly with a proper kernel
    # Recompute with cleaner approach
    
    # Clear previous attempt and start fresh
    batch_sums = torch.empty(batch_size, device=predictions.device, dtype=torch.float32)
    
    # Use a more straightforward reduction approach
    @triton.jit
    def reduce_batch_kernel(
        in_ptr,
        out_ptr,
        batch_size,
        elements_per_batch,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        
        # Accumulate sum for this batch sample
        sum_val = tl.zeros([1], dtype=tl.float32)
        
        # Process in chunks
        for i in range(0, elements_per_batch, 128):
            offsets = batch_idx * elements_per_batch + i + tl.arange(0, 128)
            mask = offsets < (batch_idx + 1) * elements_per_batch
            x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
            sum_val += tl.sum(x, axis=0)
        
        tl.store(out_ptr + batch_idx, sum_val)
    
    # Launch the reduction kernel
    reduce_batch_kernel[(batch_size,)](
        kl_intermediate, batch_sums,
        batch_size, elements_per_sample,
        BLOCK_SIZE=BLOCK_REDUCE
    )
    
    # Final result: mean over batch
    result = torch.mean(batch_sums)
    
    return result


class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)