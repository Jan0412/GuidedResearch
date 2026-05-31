import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(
    x_ptr, 
    out_ptr, 
    stride_0, stride_1, stride_2, 
    dim_0, dim_1, dim_2, 
    red_dim, 
    BLOCK_SIZE: tl.constexpr,
):
    # Map program ID to the indices of the output tensor
    pid = tl.program_id(0)
    
    if red_dim == 0:
        # Reduce dimension 0: Output shape (dim_1, dim_2)
        i = pid // dim_2
        j = pid % dim_2
        base_ptr = x_ptr + i * stride_1 + j * stride_2
        red_stride = stride_0
        dim_red = dim_0
    elif red_dim == 1:
        # Reduce dimension 1: Output shape (dim_0, dim_2)
        i = pid // dim_2
        j = pid % dim_2
        base_ptr = x_ptr + i * stride_0 + j * stride_2
        red_stride = stride_1
        dim_red = dim_1
    else: # red_dim == 2
        # Reduce dimension 2: Output shape (dim_0, dim_1)
        i = pid // dim_1
        j = pid % dim_1
        base_ptr = x_ptr + i * stride_0 + j * stride_1
        red_stride = stride_2
        dim_red = dim_2

    max_val = -float('inf')
    max_idx = 0

    # Loop over the reduction dimension in blocks
    for k_start in range(0, dim_red, BLOCK_SIZE):
        offsets = k_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_red
        
        # Load a block of values along the reduction dimension
        vals = tl.load(base_ptr + offsets * red_stride, mask=mask, other=-float('inf'))
        
        # Find max and argmax within the block
        local_max = tl.max(vals, axis=0)
        local_idx = tl.argmax(vals, axis=0)
        
        # Update global max and index if the local max is strictly greater
        # Since we are reducing a single line per program, these are scalars
        if local_max > max_val:
            max_val = local_max
            max_idx = k_start + local_idx

    # Store the final argmax index in the output tensor
    # Output is contiguous, so we can use pid directly
    tl.store(out_ptr + pid, max_idx.to(tl.int64))

def triton_argmax(x: torch.Tensor, dim: int):
    """
    Triton wrapper for torch.argmax.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure input is contiguous to simplify stride calculations
    x = x.contiguous()
    shape = x.shape
    strides = x.stride()
    
    # Calculate output shape
    out_shape = list(shape)
    del out_shape[dim]
    
    # Prepare output tensor (argmax returns int64)
    out = torch.empty(out_shape, dtype=torch.int64, device=x.device)
    
    # Total number of reductions (one per program)
    num_reductions = out.numel()
    
    # Block size for the reduction dimension
    BLOCK_SIZE = 1024
    
    # Launch the kernel
    grid = (num_reductions,)
    argmax_kernel[grid](
        x, out, 
        strides[0], strides[1], strides[2], 
        shape[0], shape[1], shape[2], 
        dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension using Triton.
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
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        # Use custom Triton kernel for argmax
        return triton_argmax(x, self.dim)