import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer (indices)
    n_elements,  # Total number of elements in input
    dim_size,  # Size of the dimension we're taking argmax over
    other_dims_product,  # Product of all other dimensions
    BLOCK_SIZE: tl.constexpr,
    DIM_SIZE: tl.constexpr,
):
    # Each program handles one output element (one index per "row")
    batch_id = tl.program_id(0)
    
    # Calculate starting offset for this batch
    start_offset = batch_id * dim_size
    
    # Initialize max value and index
    max_val = tl.full([1], float("-inf"), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Process in blocks
    for start in range(0, dim_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load values
        ptr = x_ptr + start_offset + offsets
        vals = tl.load(ptr, mask=mask, other=float("-inf"))
        
        # Update max if needed
        greater_mask = vals > max_val
        max_val = tl.where(greater_mask, vals, max_val)
        max_idx = tl.where(greater_mask, offsets, max_idx)
    
    # Store result
    tl.store(out_ptr + batch_id, max_idx)


class TritonArgmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Calculate output shape
        shape = list(x.shape)
        dim_size = shape[dim]
        shape.pop(dim)
        output_size = 1
        for s in shape:
            output_size *= s
            
        # Create output tensor
        out = torch.empty(shape, dtype=torch.long, device=x.device)
        
        # Configure kernel launch
        BLOCK_SIZE = 128  # Tunable parameter
        grid = (output_size,)
        
        # Launch kernel
        argmax_kernel[grid](
            x, out, x.numel(), dim_size, output_size,
            BLOCK_SIZE=BLOCK_SIZE, DIM_SIZE=dim_size
        )
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # Argmax is not differentiable in the traditional sense,
        # so we return None for gradients
        return None, None


def triton_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply argmax operation using Triton kernel.
    """
    return TritonArgmaxFunction.apply(x, dim)


class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension using Triton kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return triton_argmax(x, self.dim)