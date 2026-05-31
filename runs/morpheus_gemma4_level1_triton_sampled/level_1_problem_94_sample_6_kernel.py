import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_kernel(
    x_ptr, 
    y_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID to determine the block of data this instance handles
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to handle cases where n_elements is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared difference: (x - y)^2
    diff = x - y
    sq = diff * diff
    
    # Sum the squared differences within the block
    # Since we loaded out-of-bounds elements as 0.0, they don't affect the sum
    block_sum = tl.sum(sq, axis=0)
    
    # Store the partial sum in the output buffer
    tl.store(out_ptr + pid, block_sum)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Custom Triton implementation of Mean Squared Error.
    Fuses subtraction, squaring, and partial reduction into one kernel.
    """
    # Ensure tensors are contiguous and on the same GPU device
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    # Large block size to maximize GPU utilization and reduce number of partial sums
    BLOCK_SIZE = 4096
    
    # Calculate grid size (number of blocks)
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create a buffer to store partial sums from each block
    partial_sums = torch.empty(grid, device=predictions.device, dtype=torch.float32)
    
    # Launch the Triton kernel
    mse_kernel[grid](
        predictions, 
        targets, 
        partial_sums, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Perform the final global reduction and calculate the mean
    # PyTorch's sum is efficient for the remaining small buffer
    total_sum = torch.sum(partial_sums)
    return total_sum / n_elements

class ModelNew(nn.Module):
    """
    An optimized model that computes the Mean Squared Error loss using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace torch.mean((predictions - targets) ** 2) with fused Triton implementation
        return triton_mse(predictions, targets)