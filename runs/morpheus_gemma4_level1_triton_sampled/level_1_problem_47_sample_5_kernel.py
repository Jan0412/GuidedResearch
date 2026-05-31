import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduce_kernel(
    x_ptr, 
    out_ptr, 
    n_elements_reduce, 
    stride_reduce, 
    stride_out_0, 
    stride_out_1, 
    S_out_1, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to perform sum reduction over a specified dimension.
    Each program handles one reduction group (one element of the output tensor).
    """
    pid = tl.program_id(0)
    
    # Map the flat program ID to indices of the two non-reduced dimensions.
    # For a 3D tensor, if we reduce dim 'd', the other two dimensions are 'd0' and 'd1'.
    idx_0 = pid // S_out_1
    idx_1 = pid % S_out_1
    
    # Calculate the base pointer for the reduction group.
    # This pointer points to the first element of the vector to be summed.
    base_ptr = x_ptr + idx_0 * stride_out_0 + idx_1 * stride_out_1
    
    # Sum over the reduction dimension using a loop to handle arbitrary sizes.
    acc = 0.0
    for i in range(0, tl.cdiv(n_elements_reduce, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_reduce
        # Load elements along the reduction dimension.
        vals = tl.load(base_ptr + offsets * stride_reduce, mask=mask, other=0.0)
        # Perform partial reduction.
        acc += tl.sum(vals, axis=0)
    
    # Store the final sum in the output tensor.
    tl.store(out_ptr + pid, acc)

def triton_sum(x: torch.Tensor, dim: int):
    """
    Wrapper function to launch the Triton sum reduction kernel.
    """
    if not x.is_cuda:
        return torch.sum(x, dim=dim, keepdim=True)
    
    shape = x.shape
    strides = x.stride()
    
    # Normalize dimension index.
    if dim < 0:
        dim += len(shape)
        
    # Reduction dimension properties.
    n_elements_reduce = shape[dim]
    stride_reduce = strides[dim]
    
    # Identify non-reduction dimensions.
    non_reduce_dims = [i for i in range(len(shape)) if i != dim]
    
    # The current Model architecture focuses on 3D tensors (batch, dim1, dim2).
    if len(shape) == 3:
        S_out_0 = shape[non_reduce_dims[0]]
        S_out_1 = shape[non_reduce_dims[1]]
        stride_out_0 = strides[non_reduce_dims[0]]
        stride_out_1 = strides[non_reduce_dims[1]]
        
        n_out = S_out_0 * S_out_1
        # Create a flat output tensor to simplify indexing in the kernel.
        flat_out = torch.empty(n_out, device=x.device, dtype=x.dtype)
        
        # BLOCK_SIZE is chosen to balance occupancy and memory coalescing.
        BLOCK_SIZE = 1024
        grid = (n_out,)
        
        sum_reduce_kernel[grid](
            x, flat_out, 
            n_elements_reduce, stride_reduce, 
            stride_out_0, stride_out_1, S_out_1, 
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape the flat output back to the original shape with keepdim=True.
        out_shape = [shape[i] if i != dim else 1 for i in range(3)]
        return flat_out.view(*out_shape)
    else:
        # Fallback to PyTorch for tensors that are not 3D.
        return torch.sum(x, dim=dim, keepdim=True)

class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction using custom Triton kernels.
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
        Applies sum reduction over the specified dimension using Triton.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after sum reduction.
        """
        return triton_sum(x, self.dim)