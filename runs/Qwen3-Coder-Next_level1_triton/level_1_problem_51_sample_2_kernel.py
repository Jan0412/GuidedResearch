import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements in input
    dim_size,  # Size of the dimension we're taking argmax over
    other_dim_size,  # Size of the other dimensions
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
):
    # Calculate which batch we're processing
    batch_idx = tl.program_id(0)
    
    # Calculate base offset for this batch
    # For a 3D tensor, if dim=1, then offsets go like [batch, :, :]
    # The stride for dimension 1 is dim2, so base offset is batch * (dim1 * dim2)
    base_offset = batch_idx * dim_size * other_dim_size
    
    # Initialize max value and index
    max_val = -float('inf')
    max_idx = 0
    
    # Iterate through the dimension we want argmax over
    for i in range(dim_size):
        # Calculate the offset for this position
        offset = base_offset + i * other_dim_size + batch_idx * 0  # batch_idx already accounts for full batch stride
        
        # Actually, for proper indexing, we need to consider the layout
        # Let's recalculate: for tensor of shape [B, D1, D2], if dim=1 (argmax over D1)
        # Then for batch b, we want to find argmax over index [b, :, :]
        # The data layout is row-major, so [b, i, j] is at offset b*(D1*D2) + i*D2 + j
        
        # For general case: if we're argmaxing over dimension 'dim', then:
        # For position (batch, i, rest), the offset calculation depends on the dimension
        
        # Let's handle the 2D case first (which is what our input is)
        # Input is [batch_size, dim1, dim2] = [128, 4096, 4095]
        # If dim=1, we're argmaxing over dim1, output is [128, 4095]
        # If dim=2, we're argmaxing over dim2, output is [128, 4096]
        
        # For simplicity, let's assume the input is always 3D and we handle the dimension properly
        
        # Actually, let's make this work for any number of dimensions by using stride calculation
        # But for our specific case, input is [batch_size, dim1, dim2] and dim is specified
    
    # Since the above approach is getting complex, let's use a simpler approach for 3D tensors
    # We'll treat it as processing each "row" along the specified dimension
    
    # For input [B, D1, D2] and dim=1 (argmax over D1):
    # Each output element out[b, j] = argmax_i x[b, i, j]
    # So for fixed b and j, we iterate i from 0 to D1-1
    
    # For input [B, D1, D2] and dim=2 (argmax over D2):
    # Each output element out[b, i] = argmax_j x[b, i, j]
    # So for fixed b and i, we iterate j from 0 to D2-1
    
    # Let's restructure: we'll have one program per output element
    # Output has shape [batch_size, other_dim_size] where other_dim_size is the size of dimensions except dim
    
    # Calculate output indices
    # For dim=1: out_idx = batch_idx * other_dim_size + j (where j is from 0 to other_dim_size-1)
    # For dim=2: out_idx = batch_idx * other_dim_size + i (where i is from 0 to other_dim_size-1)
    
    # But actually, let's make it more general by having each program handle one output element
    out_idx = batch_idx  # We'll have grid[0] = batch_size * other_dim_size
    
    # Calculate which batch and which position in the other dimensions
    batch = out_idx // other_dim_size
    pos_in_other = out_idx % other_dim_size
    
    # Now calculate the base offset for this position
    # For dim=1: position is [batch, :, pos_in_other], stride is [D1*D2, D2, 1]
    # For dim=2: position is [batch, pos_in_other, :], stride is [D1*D2, D2, 1]
    
    # Let's pass strides instead for generality
    # Actually, for simplicity, let's assume the tensor is contiguous and we calculate offsets
    
    # For our specific case with 3D tensor [B, D1, D2]:
    if DIM == 1:
        # We want argmax over D1, output shape [B, D2]
        # For output position [batch, pos_in_other], we're looking at [batch, :, pos_in_other]
        # Offsets are: batch*(D1*D2) + i*D2 + pos_in_other for i in [0, D1-1]
        base = batch * dim_size * other_dim_size + pos_in_other
        stride = other_dim_size  # stride for dimension 1
    else:  # DIM == 2
        # We want argmax over D2, output shape [B, D1]
        # For output position [batch, pos_in_other], we're looking at [batch, pos_in_other, :]
        # Offsets are: batch*(D1*D2) + pos_in_other*D2 + i for i in [0, D2-1]
        base = batch * dim_size * other_dim_size + pos_in_other * other_dim_size
        stride = 1  # stride for dimension 2
    
    # Initialize max
    max_val = -float('inf')
    max_idx = 0
    
    # Iterate through the dimension
    for i in range(dim_size):
        offset = base + i * stride
        val = tl.load(x_ptr + offset)
        if val > max_val:
            max_val = val
            max_idx = i
    
    # Store the result
    tl.store(out_ptr + out_idx, max_idx)


def triton_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes argmax along the specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute argmax
        
    Returns:
        Tensor with argmax applied, with the specified dimension removed
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor dimensions
    shape = list(x.shape)
    dim_size = shape[dim]
    other_dim_size = 1
    for i, s in enumerate(shape):
        if i != dim:
            other_dim_size *= s
    
    # Calculate output shape
    output_shape = shape.copy()
    del output_shape[dim]
    
    # Prepare output tensor
    out = torch.empty(output_shape, dtype=torch.long, device=x.device)
    
    # Grid configuration: one program per output element
    grid = (other_dim_size,)  # We'll handle batches in the kernel
    
    # Launch kernel
    argmax_kernel[grid](
        x, out, x.numel(), dim_size, other_dim_size,
        BLOCK_SIZE=128, DIM=dim
    )
    return out


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