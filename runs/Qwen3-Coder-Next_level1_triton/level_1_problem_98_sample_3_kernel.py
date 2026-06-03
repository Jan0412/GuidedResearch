import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def kl_div_kernel(
    p_ptr,  # Pointer to predictions (P)
    q_ptr,  # Pointer to targets (Q)
    out_ptr,  # Pointer to output
    n_rows,  # Number of rows (batch size)
    n_cols,  # Number of columns (features)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one sample in batch)
    row_idx = tl.program_id(0)
    
    # Offset to start of this row
    p_row_start = row_idx * n_cols
    q_row_start = row_idx * n_cols
    out_row_start = row_idx * n_cols
    
    # Accumulator for KL divergence of this row
    kl_sum = tl.zeros((1,), dtype=tl.float32)
    
    # Process in blocks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_indices = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_indices < n_cols
        
        # Load P and Q values
        p = tl.load(p_ptr + p_row_start + col_indices, mask=mask, other=0.0)
        q = tl.load(q_ptr + q_row_start + col_indices, mask=mask, other=0.0)
        
        # Compute KL: P * (log(P) - log(Q)) = P * log(P/Q)
        # Use safe computation: P * (log(P) - log(Q))
        # Avoid log(0) by using where
        valid_mask = mask & (q > 0)
        
        # Compute log(P/Q) = log(P) - log(Q)
        log_p = tl.log(p)
        log_q = tl.log(q)
        log_ratio = log_p - log_q
        
        # Compute contribution: P * log_ratio, but only where q > 0
        contribution = p * log_ratio
        kl_sum += tl.sum(contribution * valid_mask)
    
    # Store the result for this row
    tl.store(out_ptr + row_idx, kl_sum)


@triton.jit
def batchmean_kernel(
    partial_sums_ptr,  # Pointer to row-wise KL sums
    out_ptr,  # Pointer to final output (single value)
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    # Accumulate all row sums
    total = tl.zeros((1,), dtype=tl.float32)
    
    for row_offset in range(0, n_rows, BLOCK_SIZE):
        row_indices = row_offset + tl.arange(0, BLOCK_SIZE)
        mask = row_indices < n_rows
        
        partial = tl.load(partial_sums_ptr + row_indices, mask=mask, other=0.0)
        total += tl.sum(partial)
    
    # Compute mean: divide by n_rows
    mean = total / n_rows
    tl.store(out_ptr, mean)


def triton_kl_div(predictions, targets):
    """
    Triton implementation of KL divergence with batchmean reduction.
    
    Args:
        predictions: P tensor of shape (batch_size, num_features)
        targets: Q tensor of shape (batch_size, num_features)
    
    Returns:
        Scalar tensor with mean KL divergence across batch
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size, num_features = predictions.shape
    
    # Allocate intermediate storage for row-wise KL sums
    partial_sums = torch.empty(batch_size, dtype=torch.float32, device='cuda')
    # Allocate output (scalar)
    output = torch.empty(1, dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = 256
    
    # Launch kernel for each row
    grid = (batch_size,)
    kl_div_kernel[grid](
        predictions, targets, partial_sums,
        batch_size, num_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute mean of partial sums
    grid = (1,)
    batchmean_kernel[grid](
        partial_sums, output,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output[0]


class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using Triton kernels.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use Triton implementation of KL divergence with batchmean reduction
        return triton_kl_div(predictions, targets)