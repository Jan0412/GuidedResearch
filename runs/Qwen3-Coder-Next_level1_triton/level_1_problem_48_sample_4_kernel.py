import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the number of reduction groups (one per output element)
    group_id = tl.program_id(0)
    # Compute which row/batch we're working on
    batch_idx = group_id // dim_size
    col_idx = group_id % dim_size
    
    # Calculate the starting pointer for this group's data
    # We're reducing over dimension 1 (dim1), so for each (batch_idx, col_idx),
    # we need to reduce over dim1 elements
    # Note: This assumes we're reducing over dim=1, which corresponds to dim1 dimension
    # For a 3D tensor with shape (batch_size, dim1, dim2), reducing over dim=1 means
    # we process along the dim1 dimension, resulting in (batch_size, dim2) output
    
    # However, the problem states reducing over a specific dimension, and the model
    # stores this in self.dim. We'll handle this more generally in the wrapper
    
    # For now, let's assume we're reducing over dim=1 (dim1), so:
    # Output shape: (batch_size, dim2)
    # Input shape: (batch_size, dim1, dim2)
    # For output index [batch_idx, col_idx], we need to sum over dim1
    
    # Compute the base offset for this output element
    # In row-major layout: offset = batch_idx * (dim1 * dim2) + col_idx
    base_offset = batch_idx * (dim_size * (n_elements // (batch_size * dim_size))) + col_idx
    
    # Sum over the reduction dimension
    sum_val = 0.0
    count = 0
    
    # We'll iterate over the reduction dimension in blocks
    # For reduction over dim1, the stride between consecutive elements in that dim is 1
    # But we need to skip by dim1 each iteration
    for i in range(0, dim_size, BLOCK_SIZE):
        offsets = base_offset + i + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid elements
        mask = (i + tl.arange(0, BLOCK_SIZE)) < dim_size
        # Load with mask
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        # Accumulate sum
        sum_val += tl.sum(x, axis=0)
        count += BLOCK_SIZE if (i + BLOCK_SIZE) <= dim_size else (dim_size - i)
    
    # Compute mean
    mean_val = sum_val / count
    
    # Store result
    tl.store(out_ptr + group_id, mean_val)


@triton.jit
def mean_dim0_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim0_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Reducing over dimension 0 (batch dimension)
    # Output shape: (dim1, dim2)
    # Input shape: (batch_size, dim1, dim2)
    # For output index [i, j], we sum over batch dimension
    
    out_idx = tl.program_id(0)
    
    # Compute i, j from out_idx (flattened output index)
    i = out_idx // (n_elements // dim0_size)
    j = out_idx % (n_elements // dim0_size)
    
    # Base offset for [i,j] in the output
    base_offset = i * (n_elements // dim0_size) + j
    
    sum_val = 0.0
    count = 0
    
    for k in range(0, dim0_size, BLOCK_SIZE):
        # For each batch element k, the offset in the input is:
        # k * (dim1 * dim2) + i * dim2 + j
        offsets = k * (n_elements // dim0_size) + base_offset
        # Load single element (using scalar load)
        x = tl.load(x_ptr + offsets)
        sum_val += x
        count += 1
    
    mean_val = sum_val / count
    tl.store(out_ptr + out_idx, mean_val)


@triton.jit
def mean_dim2_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim2_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Reducing over dimension 2 (last dimension)
    # Input shape: (batch_size, dim1, dim2)
    # Output shape: (batch_size, dim1)
    # For output index [i, j], we sum over dim2 dimension
    
    out_idx = tl.program_id(0)
    
    # Compute i, j from out_idx
    i = out_idx // (n_elements // (dim2_size * (n_elements // dim2_size)))
    j = out_idx % (n_elements // (dim2_size * (n_elements // dim2_size)))
    
    # For each [i,j] pair, the base offset in input is:
    # i * (dim1 * dim2) + j * dim2
    base_offset = i * (n_elements // (n_elements // dim2_size)) + j * dim2_size
    
    sum_val = 0.0
    count = 0
    
    for k in range(0, dim2_size, BLOCK_SIZE):
        offsets = base_offset + k + tl.arange(0, BLOCK_SIZE)
        mask = (k + tl.arange(0, BLOCK_SIZE)) < dim2_size
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        count += BLOCK_SIZE if (k + BLOCK_SIZE) <= dim2_size else (dim2_size - k)
    
    mean_val = sum_val / count
    tl.store(out_ptr + out_idx, mean_val)


def triton_mean(x: torch.Tensor, dim: int):
    """
    Compute mean along specified dimension using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    shape = x.shape
    n_elements = x.numel()
    
    # Determine output shape
    if dim < 0:
        dim = len(shape) + dim
    
    # Calculate output shape and size
    out_shape = list(shape)
    out_shape.pop(dim)
    out_size = 1
    for s in out_shape:
        out_size *= s
    
    # Create output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = 256
    
    if dim == 0:
        # Reduce over batch dimension
        dim0_size = shape[0]
        grid = (out_size,)
        mean_dim0_kernel[grid](x, out, n_elements, dim0_size, BLOCK_SIZE=BLOCK_SIZE)
    elif dim == 1:
        # Reduce over dim1 (second dimension)
        dim1_size = shape[1]
        # Output has shape (batch_size, dim2) = (shape[0], shape[2])
        grid = (shape[0] * shape[2],)
        mean_kernel[grid](x, out, n_elements, dim1_size, BLOCK_SIZE=BLOCK_SIZE)
    elif dim == 2:
        # Reduce over last dimension
        dim2_size = shape[2]
        grid = (shape[0] * shape[1],)
        mean_dim2_kernel[grid](x, out, n_elements, dim2_size, BLOCK_SIZE=BLOCK_SIZE)
    else:
        raise ValueError(f"Dimension {dim} not supported for tensor of shape {shape}")
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension using Triton kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        return triton_mean(x, self.dim)