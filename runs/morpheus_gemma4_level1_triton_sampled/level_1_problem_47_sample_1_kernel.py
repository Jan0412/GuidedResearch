import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduce_kernel(
    x_ptr, 
    out_ptr, 
    B, D1, D2, 
    S_B, S_D1, S_D2, 
    S_OB, S_OD2,
    BLOCK_D1: tl.constexpr, 
    BLOCK_D2: tl.constexpr,
):
    # Each program handles a block of (B, D2)
    pid = tl.program_id(0)
    
    # Calculate the number of blocks along D2
    num_d2_blocks = (D2 + BLOCK_D2 - 1) // BLOCK_D2
    
    # Map pid to batch index b and d2 start index
    b = pid // num_d2_blocks
    d2_idx = (pid % num_d2_blocks) * BLOCK_D2
    
    # Create offsets for the D2 dimension
    d2_offsets = d2_idx + tl.arange(0, BLOCK_D2)
    mask_d2 = d2_offsets < D2
    
    # Initialize accumulator for the reduction
    acc = tl.zeros([BLOCK_D2], dtype=tl.float32)
    
    # Loop over the reduction dimension D1 in blocks
    for i in range(0, D1, BLOCK_D1):
        d1_offsets = i + tl.arange(0, BLOCK_D1)
        mask_d1 = d1_offsets < D1
        
        # Compute pointers for the block (BLOCK_D1, BLOCK_D2)
        # Pointer = base + b*S_B + d1_off*S_D1 + d2_off*S_D2
        # We use broadcasting to create a 2D grid of pointers
        ptr = x_ptr + b * S_B + d1_offsets[:, None] * S_D1 + d2_offsets[None, :] * S_D2
        
        # Load the block and sum along the D1 axis
        vals = tl.load(ptr, mask=mask_d1[:, None] & mask_d2[None, :], other=0.0)
        acc += tl.sum(vals, axis=0)
        
    # Compute the output pointer for the result
    # Output shape is (B, 1, D2)
    out_ptr_base = out_ptr + b * S_OB + d2_offsets * S_OD2
    tl.store(out_ptr_base, acc, mask=mask_d2)

def triton_sum(x: torch.Tensor, dim: int):
    """
    Triton wrapper for sum reduction over a specific dimension.
    Optimized for 3D tensors reducing over dim=1.
    """
    # This specific implementation is optimized for 3D tensors reducing over dim=1
    # as per the provided architecture and input parameters.
    assert x.ndim == 3 and dim == 1, "This kernel is optimized for 3D tensors and dim=1."
    
    B, D1, D2 = x.shape
    x = x.contiguous()
    
    # Prepare output tensor (B, 1, D2)
    out = torch.empty((B, 1, D2), device=x.device, dtype=x.dtype)
    
    # Strides
    S_B, S_D1, S_D2 = x.stride()
    S_OB, S_OD1, S_OD2 = out.stride()
    
    # Tuning parameters
    BLOCK_D1 = 1024
    BLOCK_D2 = 32
    
    # Grid: one program for every (B, block of D2)
    num_d2_blocks = (D2 + BLOCK_D2 - 1) // BLOCK_D2
    grid = (B * num_d2_blocks,)
    
    sum_reduce_kernel[grid](
        x, out, 
        B, D1, D2, 
        S_B, S_D1, S_D2, 
        S_OB, S_OD2,
        BLOCK_D1=BLOCK_D1, 
        BLOCK_D2=BLOCK_D2
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
        # Use the optimized Triton kernel for the target case (3D tensor, dim=1)
        if x.ndim == 3 and self.dim == 1:
            return triton_sum(x, self.dim)
        else:
            # Fallback to PyTorch for other configurations
            return torch.sum(x, dim=self.dim, keepdim=True)