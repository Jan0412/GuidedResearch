import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduction_kernel(
    x_ptr, 
    out_ptr, 
    S_reduce, 
    S_other0, 
    S_other1, 
    stride_reduce, 
    stride_other0, 
    stride_other1, 
    out_stride_other0, 
    out_stride_other1, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one reduction line across the specified dimension
    pid = tl.program_id(0)
    
    # Map the program ID to the indices of the non-reduction dimensions
    idx0 = pid // S_other1
    idx1 = pid % S_other1
    
    # Pointer to the start of the reduction line for the current (idx0, idx1)
    # This represents x[idx0, 0, idx1] if dim=1
    ptr = x_ptr + idx0 * stride_other0 + idx1 * stride_other1
    
    acc = 0.0
    # Loop over the reduction dimension in blocks of BLOCK_SIZE
    for i in range(0, S_reduce, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < S_reduce
        # Load elements along the reduction dimension
        vals = tl.load(ptr + offsets * stride_reduce, mask=mask, other=0.0)
        # Sum within the block and accumulate
        acc += tl.sum(vals, axis=0)
    
    # Calculate the pointer to the output element in the reduced tensor
    out_ptr_final = out_ptr + idx0 * out_stride_other0 + idx1 * out_stride_other1
    tl.store(out_ptr_final, acc)

def triton_sum(x: torch.Tensor, dim: int):
    """
    Triton wrapper for torch.sum(x, dim=dim, keepdim=True)
    Optimized for 3D tensors.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    shape = x.shape
    strides = x.stride()
    ndim = x.ndim
    
    # Identify reduction dimension and the other two dimensions
    S_reduce = shape[dim]
    stride_reduce = strides[dim]
    
    others = [i for i in range(ndim) if i != dim]
    S_other0 = shape[others[0]]
    S_other1 = shape[others[1]]
    stride_other0 = strides[others[0]]
    stride_other1 = strides[others[1]]
    
    # Prepare output tensor with keepdim=True
    out_shape = [shape[i] if i != dim else 1 for i in range(ndim)]
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    out_strides = out.stride()
    out_stride_other0 = out_strides[others[0]]
    out_stride_other1 = out_strides[others[1]]
    
    # Grid: one program per reduction line (B * D2 for dim=1)
    grid = (S_other0 * S_other1,)
    # Block size for reduction; 1024 is a good balance for most GPUs
    BLOCK_SIZE = 1024
    
    sum_reduction_kernel[grid](
        x, out, 
        S_reduce, S_other0, S_other1, 
        stride_reduce, stride_other0, stride_other1, 
        out_stride_other0, out_stride_other1, 
        BLOCK_SIZE=BLOCK_SIZE
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
        Applies sum reduction over the specified dimension using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return triton_sum(x, self.dim)