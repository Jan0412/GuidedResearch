import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduction_kernel(
    x_ptr, 
    out_ptr, 
    stride_b, 
    stride_red, 
    stride_out, 
    out_stride_b, 
    out_stride_out, 
    B, 
    D_red, 
    D_out, 
    BLOCK_SIZE_K: tl.constexpr, 
    BLOCK_SIZE_RED: tl.constexpr,
):
    # Program ID for the batch and the output dimension (K)
    pid = tl.program_id(0)
    num_pids_k = tl.cdiv(D_out, BLOCK_SIZE_K)
    pid_b = pid // num_pids_k
    pid_k = pid % num_pids_k

    # Compute offsets for the output dimension
    k_offsets = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask_k = k_offsets < D_out

    # Initialize accumulator for the reduction
    acc = tl.zeros([BLOCK_SIZE_K], dtype=tl.float32)

    # Loop over the reduction dimension in blocks
    for j in range(0, D_red, BLOCK_SIZE_RED):
        j_offsets = j + tl.arange(0, BLOCK_SIZE_RED)
        mask_j = j_offsets < D_red
        
        # Compute 2D offsets: (BLOCK_SIZE_RED, BLOCK_SIZE_K)
        # x[pid_b, j, k]
        offsets = (
            pid_b * stride_b + 
            j_offsets[:, None] * stride_red + 
            k_offsets[None, :] * stride_out
        )
        
        # Load a block of data and sum along the reduction axis (axis 0)
        vals = tl.load(x_ptr + offsets, mask=mask_j[:, None] & mask_k[None, :], other=0.0)
        acc += tl.sum(vals, axis=0)

    # Compute output offsets and store the result
    out_offsets = pid_b * out_stride_b + k_offsets * out_stride_out
    tl.store(out_ptr + out_offsets, acc, mask=mask_k)

def triton_sum(x: torch.Tensor, dim: int):
    """
    Triton wrapper for sum reduction over a specific dimension.
    Optimized for the case where dim=1 in a 3D tensor (B, D_red, D_out).
    """
    # This specific implementation targets the (B, D_red, D_out) case with dim=1
    # as requested by the architecture provided.
    assert x.dim() == 3 and dim == 1, "This kernel is optimized for 3D tensors and dim=1"
    
    B, D_red, D_out = x.shape
    stride_b, stride_red, stride_out = x.stride()
    
    # Prepare output tensor with keepdim=True
    out = torch.empty((B, 1, D_out), device=x.device, dtype=x.dtype)
    out_stride_b, _, out_stride_out = out.stride()

    # Tunable parameters
    BLOCK_SIZE_K = 256
    BLOCK_SIZE_RED = 1024

    # Grid: one program per block of the output dimension for each batch element
    num_pids_k = triton.cdiv(D_out, BLOCK_SIZE_K)
    grid = (B * num_pids_k,)

    sum_reduction_kernel[grid](
        x, out, 
        stride_b, stride_red, stride_out, 
        out_stride_b, out_stride_out, 
        B, D_red, D_out, 
        BLOCK_SIZE_K=BLOCK_SIZE_K, 
        BLOCK_SIZE_RED=BLOCK_SIZE_RED
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
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        # Ensure input is on CUDA and contiguous for the kernel
        if not x.is_cuda:
            return torch.sum(x, dim=self.dim, keepdim=True)
        
        # For the specific dimensions provided (B=128, D1=4096, D2=4095, dim=1),
        # we use the custom Triton kernel.
        return triton_sum(x, dim=self.dim)