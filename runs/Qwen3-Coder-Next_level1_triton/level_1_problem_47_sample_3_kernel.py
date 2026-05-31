import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduction_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements in input
    stride_dim,  # Stride along the reduction dimension
    dim_size,  # Size of the reduction dimension
    BLOCK_SIZE: tl.constexpr,
    REDUCTION_BLOCK_SIZE: tl.constexpr,
):
    # Get the program id for the outer dimensions (non-reduction dimensions)
    outer_idx = tl.program_id(0)
    
    # Calculate the base offset for this outer index
    # For a 3D tensor (batch, dim1, dim2) reducing over dim1:
    # outer_idx corresponds to (batch_idx, dim2_idx) combined
    batch_idx = outer_idx // dim_size
    dim2_idx = outer_idx % dim_size
    
    # Calculate base offset in the input tensor
    base_offset = batch_idx * stride_dim * dim_size + dim2_idx
    
    # Accumulator for the sum
    sum_result = tl.zeros([REDUCTION_BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over the reduction dimension in chunks
    for start in range(0, dim_size, REDUCTION_BLOCK_SIZE):
        # Calculate offsets within the reduction dimension
        offsets = start + tl.arange(0, REDUCTION_BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Calculate the actual memory offsets for this chunk
        mem_offsets = base_offset + offsets * stride_dim
        
        # Load the values
        vals = tl.load(x_ptr + mem_offsets, mask=mask, other=0.0)
        
        # Convert to float32 for accumulation (to match PyTorch default behavior)
        vals_f32 = vals.to(tl.float32)
        
        # Accumulate
        sum_result = sum_result + vals_f32
    
    # Final reduction within the block
    for i in range(REDUCTION_BLOCK_SIZE // 2):
        if i == 0:
            temp_sum = sum_result
        temp_sum = temp_sum + tl.roll(temp_sum, 1)
    
    # Store the final result
    if tl.program_id(0) < n_elements // dim_size:
        # Write the reduced value to the output
        out_offset = outer_idx
        tl.store(out_ptr + out_offset, sum_result[0])


def triton_sum_reduction(x: torch.Tensor, dim: int):
    """
    Perform sum reduction over the specified dimension using Triton.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce over
        
    Returns:
        Output tensor with the specified dimension reduced (keepdim=True)
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shape and calculate necessary parameters
    shape = x.shape
    ndim = len(shape)
    
    # Normalize negative dimension index
    if dim < 0:
        dim += ndim
    
    # Calculate the stride for the reduction dimension
    # For a tensor with shape (d0, d1, d2, ...), stride[i] = product of shapes[i+1:]
    stride = 1
    for i in range(ndim - 1, dim, -1):
        stride *= shape[i]
    
    # Calculate output shape
    out_shape = list(shape)
    out_shape[dim] = 1
    out_shape = tuple(out_shape)
    
    # Calculate number of outer elements (elements not in reduction dimension)
    n_outer = 1
    for i, s in enumerate(shape):
        if i != dim:
            n_outer *= s
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Configuration for the kernel
    BLOCK_SIZE = 1  # For outer dimension parallelism
    REDUCTION_BLOCK_SIZE = 256  # Size for reduction within each block
    
    # Grid configuration: one block per outer element
    grid = (n_outer,)
    
    # Launch the kernel
    sum_reduction_kernel[grid](
        x, out, n_outer, stride, shape[dim],
        BLOCK_SIZE=BLOCK_SIZE,
        REDUCTION_BLOCK_SIZE=REDUCTION_BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension using Triton.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return triton_sum_reduction(x, self.dim)