import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduction_kernel(
    x_ptr, 
    out_ptr,
    S_red, 
    S_o0, S_o1,
    stride_red, stride_o0, stride_o1,
    out_stride_o0, out_stride_o1,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one reduction group (one "line" of the tensor)
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    
    # Calculate the starting pointer for the reduction line
    # The line is defined by the two non-reduction dimensions
    ptr = x_ptr + pid_0 * stride_o0 + pid_1 * stride_o1
    
    acc = 0.0
    i = 0
    # Loop over the reduction dimension in blocks
    while i < S_red:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < S_red
        # Load elements along the reduction dimension
        vals = tl.load(ptr + offsets * stride_red, mask=mask, other=0.0)
        # Sum the current block and accumulate
        acc += tl.sum(vals, axis=0)
        i += BLOCK_SIZE
        
    # Calculate the output pointer for the result
    out_ptr_final = out_ptr + pid_0 * out_stride_o0 + pid_1 * out_stride_o1
    tl.store(out_ptr_final, acc)

def triton_sum(x: torch.Tensor, dim: int):
    """
    Triton wrapper for sum reduction over a specified dimension for a 3D tensor.
    """
    # Ensure input is contiguous to simplify pointer arithmetic
    x = x.contiguous()
    S = x.shape
    stride = x.stride()
    
    # Determine which dimension is being reduced and identify the others
    if dim == 0:
        S_red = S[0]
        S_o0, S_o1 = S[1], S[2]
        stride_red = stride[0]
        stride_o0, stride_o1 = stride[1], stride[2]
    elif dim == 1:
        S_red = S[1]
        S_o0, S_o1 = S[0], S[2]
        stride_red = stride[1]
        stride_o0, stride_o1 = stride[0], stride[2]
    elif dim == 2:
        S_red = S[2]
        S_o0, S_o1 = S[0], S[1]
        stride_red = stride[2]
        stride_o0, stride_o1 = stride[0], stride[1]
    else:
        raise ValueError("Only dimensions 0, 1, 2 are supported for this 3D sum kernel.")
        
    # Prepare output tensor with keepdim=True
    out_shape = list(S)
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    out_stride = out.stride()
    
    # Map output strides to the grid's o0, o1 order
    if dim == 0:
        os0, os1 = out_stride[1], out_stride[2]
    elif dim == 1:
        os0, os1 = out_stride[0], out_stride[2]
    else: # dim == 2
        os0, os1 = out_stride[0], out_stride[1]
        
    # Grid is based on the two dimensions not being reduced
    grid = (S_o0, S_o1)
    
    # Launch the kernel
    sum_reduction_kernel[grid](
        x, out, 
        S_red, 
        S_o0, S_o1, 
        stride_red, stride_o0, stride_o1, 
        os0, os1, 
        BLOCK_SIZE=1024
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
        # Triton kernels require CUDA tensors
        if not x.is_cuda:
            return torch.sum(x, dim=self.dim, keepdim=True)
        
        return triton_sum(x, self.dim)