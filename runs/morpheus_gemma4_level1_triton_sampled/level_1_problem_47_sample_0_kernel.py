import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduction_kernel(
    x_ptr, 
    out_ptr, 
    stride_x_b, 
    stride_x_d1, 
    stride_x_d2, 
    stride_out_b, 
    stride_out_d2, 
    B, 
    D1, 
    D2, 
    BLOCK_SIZE_D1: tl.constexpr, 
    BLOCK_SIZE_D2: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_d2 = tl.program_id(1)

    # Range of indices for the D2 dimension
    offsets_d2 = pid_d2 * BLOCK_SIZE_D2 + tl.arange(0, BLOCK_SIZE_D2)
    mask_d2 = offsets_d2 < D2

    # Initialize accumulator for the sum reduction
    acc = tl.zeros([BLOCK_SIZE_D2], dtype=tl.float32)

    # Loop over the reduction dimension D1 in blocks
    for k in range(0, D1, BLOCK_SIZE_D1):
        offsets_d1 = k + tl.arange(0, BLOCK_SIZE_D1)
        mask_d1 = offsets_d1 < D1

        # Calculate pointers for the current block
        # x shape is (B, D1, D2)
        # Pointer = base + b * stride_b + d1 * stride_d1 + d2 * stride_d2
        ptr = x_ptr + pid_b * stride_x_b + offsets_d1[:, None] * stride_x_d1 + offsets_d2[None, :] * stride_x_d2
        
        # Load the block and mask out-of-bounds elements
        vals = tl.load(ptr, mask=mask_d1[:, None] & mask_d2[None, :], other=0.0)
        
        # Sum along the D1 axis (axis 0 of the loaded block)
        acc += tl.sum(vals, axis=0)

    # Store the final accumulated sum into the output tensor
    # out shape is (B, 1, D2)
    out_ptr_final = out_ptr + pid_b * stride_out_b + offsets_d2 * stride_out_d2
    tl.store(out_ptr_final, acc, mask=mask_d2)

def triton_sum_dim1(x: torch.Tensor):
    """
    Triton wrapper for sum reduction along dimension 1.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    
    # Ensure tensor is contiguous for predictable stride calculations
    x = x.contiguous()
    B, D1, D2 = x.shape
    
    # Output shape (B, 1, D2)
    out = torch.empty((B, 1, D2), device=x.device, dtype=x.dtype)
    
    # Strides
    stride_x_b, stride_x_d1, stride_x_d2 = x.stride()
    stride_out_b, _, stride_out_d2 = out.stride()

    # Tuning parameters
    # BLOCK_SIZE_D2 is kept small to fit into shared memory/registers
    BLOCK_SIZE_D1 = 1024
    BLOCK_SIZE_D2 = 32

    # Grid: (Batch size, number of blocks needed for D2)
    grid = (B, (D2 + BLOCK_SIZE_D2 - 1) // BLOCK_SIZE_D2)

    sum_reduction_kernel[grid](
        x, out, 
        stride_x_b, stride_x_d1, stride_x_d2, 
        stride_out_b, stride_out_d2, 
        B, D1, D2, 
        BLOCK_SIZE_D1=BLOCK_SIZE_D1, 
        BLOCK_SIZE_D2=BLOCK_SIZE_D2
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension.
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
        Optimized using a custom Triton kernel for dim=1.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        # The custom kernel is specifically optimized for dim=1 as per the problem setup.
        if self.dim == 1:
            return triton_sum_dim1(x)
        else:
            # Fallback to PyTorch for other dimensions
            return torch.sum(x, dim=self.dim, keepdim=True)