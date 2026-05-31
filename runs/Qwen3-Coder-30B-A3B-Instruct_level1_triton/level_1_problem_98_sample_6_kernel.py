import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def kl_div_kernel(
    log_probs_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and sequence indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate the starting position for this batch and sequence
    start_pos = batch_idx * seq_len + seq_idx * seq_len
    
    # Shared memory for reduction
    shared_mem = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Initialize accumulator for this thread
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Process elements in chunks
    for i in range(0, seq_len, BLOCK_SIZE):
        # Calculate actual offset
        offset = start_pos + i + tl.arange(0, BLOCK_SIZE)
        
        # Create mask to avoid out-of-bounds access
        mask = offset < start_pos + seq_len
        
        # Load data
        log_prob = tl.load(log_probs_ptr + offset, mask=mask, other=0.0)
        target = tl.load(targets_ptr + offset, mask=mask, other=0.0)
        
        # Compute KL divergence: target * (log(target) - log_prob)
        # Handle numerical stability: avoid log(0)
        safe_target = tl.where(target > 1e-38, target, 1e-38)
        safe_log_prob = tl.where(log_prob > 1e-38, log_prob, 1e-38)
        
        kl_term = safe_target * (tl.log(safe_target) - safe_log_prob)
        
        # Accumulate
        acc += tl.sum(kl_term)
        
    # Store partial sum in shared memory
    tid = tl.thread_id()
    if tid < BLOCK_SIZE:
        shared_mem[tid] = acc[0]
    tl.sync()
    
    # Reduction in shared memory
    if tid < BLOCK_SIZE // 2:
        shared_mem[tid] += shared_mem[tid + BLOCK_SIZE // 2]
    tl.sync()
    
    if tid < BLOCK_SIZE // 4:
        shared_mem[tid] += shared_mem[tid + BLOCK_SIZE // 4]
    tl.sync()
    
    # Final reduction
    if tid == 0:
        output_ptr[batch_idx] = shared_mem[0]

@triton.jit
def log_softmax_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sequence
    batch_idx = tl.program_id(0)
    
    # Calculate base offset for this sequence
    base_offset = batch_idx * seq_len
    
    # Shared memory for reduction operations
    shared_max = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    shared_sum = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # First pass: find max value in each block
    for i in range(0, seq_len, BLOCK_SIZE):
        offset = base_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offset < base_offset + seq_len
        
        # Load input values
        x = tl.load(input_ptr + offset, mask=mask, other=-float('inf'))
        
        # Find maximum
        local_max = tl.max(x)
        
        # Store in shared memory
        if tl.thread_id() < BLOCK_SIZE:
            shared_max[tl.thread_id()] = local_max
        tl.sync()
        
        # Reduce to get global max
        max_val = shared_max[0]
        for j in range(1, min(BLOCK_SIZE, seq_len - i)):
            max_val = tl.maximum(max_val, shared_max[j])
        
        # Second pass: compute log-sum-exp
        exp_x = tl.exp(x - max_val)
        sum_exp = tl.sum(exp_x)
        
        # Store results back
        if tl.thread_id() < BLOCK_SIZE:
            shared_sum[tl.thread_id()] = sum_exp
        tl.sync()
        
        # Reduce to get total sum
        total_sum = shared_sum[0]
        for j in range(1, min(BLOCK_SIZE, seq_len - i)):
            total_sum += shared_sum[j]
        
        # Final computation: log(prob) = x - max - log(sum_exp)
        final_result = x - max_val - tl.log(total_sum)
        
        # Store results
        tl.store(output_ptr + offset, final_result, mask=mask)

def triton_kl_div(log_probs, targets):
    """Custom Triton implementation of KL divergence"""
    assert log_probs.is_cuda and targets.is_cuda, "Tensors must be on CUDA"
    assert log_probs.shape == targets.shape, "Shapes must match"
    
    batch_size, seq_len = log_probs.shape
    
    # Allocate output tensor
    output = torch.empty(batch_size, dtype=torch.float32, device=log_probs.device)
    
    # Grid configuration
    grid = (batch_size, 1)
    BLOCK_SIZE = 1024
    
    # Launch kernel
    kl_div_kernel[grid](
        log_probs,
        targets,
        output,
        log_probs.numel(),
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean across batches
    return output.mean()

def triton_log_softmax(input_tensor):
    """Custom Triton implementation of log softmax"""
    assert input_tensor.is_cuda, "Tensor must be on CUDA"
    
    batch_size, seq_len = input_tensor.shape
    
    # Allocate output tensor
    output = torch.empty_like(input_tensor, dtype=torch.float32)
    
    # Grid configuration
    grid = (batch_size,)
    BLOCK_SIZE = 1024
    
    # Launch kernel
    log_softmax_kernel[grid](
        input_tensor,
        output,
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for KL divergence computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Apply log to predictions using Triton kernel
        log_predictions = triton_log_softmax(predictions)
        
        # Compute KL divergence using Triton kernel
        kl_loss = triton_kl_div(log_predictions, targets)
        
        return kl_loss

# Helper functions for compatibility with original interface
def get_inputs():
    scale = torch.rand(())
    return [(torch.rand(8192 * 2, 8192 * 2)*scale).softmax(dim=-1), 
            torch.rand(8192 * 2, 8192 * 2).softmax(dim=-1)]

def get_init_inputs():
    return []