import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_squares_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
    NUM_ITER: tl.constexpr
):
    """
    Triton kernel to compute partial sums of squares of a tensor.
    Each block processes multiple chunks of data to reduce the total number of blocks.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * NUM_ITER
    
    acc = 0.0
    for i in range(0, NUM_ITER):
        offsets = block_start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        # Square and sum the current block
        acc += tl.sum(x * x, axis=0)
        
    tl.store(out_ptr + pid, acc)

@triton.jit
def div_kernel(
    x_ptr, 
    out_ptr, 
    inv_norm, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
    NUM_ITER: tl.constexpr
):
    """
    Triton kernel to perform element-wise multiplication by the inverse norm.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * NUM_ITER
    
    for i in range(0, NUM_ITER):
        offsets = block_start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = x * inv_norm
        tl.store(out_ptr + offsets, out, mask=mask)

def triton_frobenius_norm_norm(x: torch.Tensor):
    """
    Wrapper function that leverages Triton kernels to perform Frobenius norm normalization.
    """
    orig_shape = x.shape
    # Flatten the tensor for easier processing in kernels
    x_flat = x.contiguous().view(-1)
    n_elements = x_flat.numel()
    
    # Tunable parameters for kernel execution
    BLOCK_SIZE = 1024
    NUM_ITER = 128
    elements_per_block = BLOCK_SIZE * NUM_ITER
    grid_size = (n_elements + elements_per_block - 1) // elements_per_block
    
    # Step 1: Compute partial sums of squares
    # We use a buffer to store the sum of squares from each block to maintain precision
    partial_sums = torch.empty(grid_size, device=x.device, dtype=torch.float32)
    sum_squares_kernel[grid_size](
        x_flat, partial_sums, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE, NUM_ITER=NUM_ITER
    )
    
    # Step 2: Global reduction and norm calculation
    # torch.sum on the small partial_sums tensor is very efficient
    total_sum_sq = torch.sum(partial_sums)
    norm = torch.sqrt(total_sum_sq)
    
    # Compute inverse norm to replace division with multiplication in the kernel
    inv_norm = 1.0 / norm
    
    # Step 3: Element-wise normalization
    out_flat = torch.empty_like(x_flat)
    div_kernel[grid_size](
        x_flat, out_flat, inv_norm, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE, NUM_ITER=NUM_ITER
    )
    
    return out_flat.view(orig_shape)

class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using Triton kernels.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor using custom Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        # Ensure input is in FP32 as required by the optimization target
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            
        return triton_frobenius_norm_norm(x)