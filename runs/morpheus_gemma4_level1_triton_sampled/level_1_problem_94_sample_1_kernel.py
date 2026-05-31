import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_kernel(
    x_ptr,  # Pointer to predictions
    y_ptr,  # Pointer to targets
    out_ptr,  # Pointer to partial sums
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values. 
    # Using other=0.0 ensures that padding elements contribute 0 to the sum of squares
    # because (0.0 - 0.0)^2 = 0.0.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Compute squared difference
    diff = x - y
    sq_diff = diff * diff

    # Sum the squared differences within the block
    # tl.sum reduces the tensor to a scalar for this block
    block_sum = tl.sum(sq_diff, axis=0)

    # Store the partial sum in the output tensor
    tl.store(out_ptr + pid, block_sum)


def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Computes the Mean Squared Error using a Triton kernel for the sum of squares
    and PyTorch for the final reduction and mean.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for Triton
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Optimized block size for FP32

    # Determine the number of blocks needed
    grid_x = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_x,)

    # Allocate a tensor to hold partial sums from each block
    partial_sums = torch.empty(grid_x, device=predictions.device, dtype=torch.float32)

    # Launch the Triton kernel to compute partial sums of squared differences
    mse_kernel[grid](
        predictions, 
        targets, 
        partial_sums, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Final reduction: sum the partial sums and divide by the total number of elements
    total_sum = torch.sum(partial_sums)
    return total_sum / n_elements


class ModelNew(nn.Module):
    """
    An optimized model that computes the Mean Squared Error loss using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace torch.mean((predictions - targets) ** 2) with the Triton implementation
        return triton_mse(predictions, targets)