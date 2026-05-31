import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    S_dim,          # Size of the reduction dimension
    S_after,        # Product of dimensions after the reduction dimension
    S_before_stride,# Stride of the dimension before the reduction dimension (S_dim * S_after)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one reduction group (one element of the output tensor)
    pid = tl.program_id(0)

    # Calculate the starting offset for this reduction group in the flattened input tensor
    # For contiguous tensors:
    # offset = (pid // S_after) * (S_dim * S_after) + (pid % S_after)
    offset = (pid // S_after) * S_before_stride + (pid % S_after)

    # Initialize accumulator
    acc = 0.0

    # Loop over the reduction dimension in blocks
    for i in range(0, S_dim, BLOCK_SIZE):
        # Generate offsets for the current block within the reduction dimension
        offsets = i + tl.arange(0, BLOCK_SIZE)
        # Mask to avoid out-of-bounds access
        mask = offsets < S_dim
        
        # Load values: index = base_offset + (k * stride_of_reduction_dim)
        # The stride of the reduction dimension is S_after
        x_vals = tl.load(x_ptr + offset + offsets * S_after, mask=mask, other=0.0)
        
        # Sum the block
        acc += tl.sum(x_vals, axis=0)

    # Compute mean and store
    out = acc / S_dim
    tl.store(out_ptr + pid, out)


def triton_mean(x: torch.Tensor, dim: int):
    """
    Triton implementation of torch.mean(x, dim=dim) for FP32.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensor is contiguous to simplify indexing logic
    x = x.contiguous()
    shape = x.shape
    ndim = x.ndim
    
    # Handle negative dimension
    if dim < 0:
        dim += ndim

    S_dim = shape[dim]
    
    # Product of dimensions after the reduction dimension
    S_after = 1
    for i in range(dim + 1, ndim):
        S_after *= shape[i]
        
    # Stride of the dimension immediately preceding the reduction dimension
    S_before_stride = S_dim * S_after
    
    # Total number of reduction groups (size of output tensor)
    num_groups = x.numel() // S_dim
    
    # Prepare output tensor
    out_shape = list(shape)
    del out_shape[dim]
    out = torch.empty(out_shape, device=x.device, dtype=torch.float32)
    out_flat = out.view(-1)

    # Block size for the reduction loop
    BLOCK_SIZE = 1024 if S_dim >= 1024 else triton.next_power_of_2(S_dim)

    # Launch grid: one program per output element
    grid = (num_groups,)

    mean_kernel[grid](
        x, 
        out_flat, 
        S_dim, 
        S_after, 
        S_before_stride, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension using Triton.
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
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        # Ensure input is FP32 for the kernel
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            
        return triton_mean(x, self.dim)