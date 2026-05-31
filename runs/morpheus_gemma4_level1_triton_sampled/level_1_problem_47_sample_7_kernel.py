import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduction_kernel(
    x_ptr, 
    out_ptr,
    S0, S1, S2,
    st0, st1, st2,
    dim,
    R,
    M,
    BLOCK_SIZE_R: tl.constexpr,
):
    # Each program handles one element of the output tensor
    m_idx = tl.program_id(0)
    if m_idx >= M:
        return

    # We map the linear output index m_idx back to the coordinates of the non-reduced dimensions.
    # The logic depends on which dimension is being reduced.
    if dim == 0:
        r_stride = st0
        m_S1 = S2
        i0 = m_idx // m_S1
        i1 = m_idx % m_S1
        base_off = i0 * st1 + i1 * st2
    elif dim == 1:
        r_stride = st1
        m_S1 = S2
        i0 = m_idx // m_S1
        i1 = m_idx % m_S1
        base_off = i0 * st0 + i1 * st2
    else: # dim == 2
        r_stride = st2
        m_S1 = S1
        i0 = m_idx // m_S1
        i1 = m_idx % m_S1
        base_off = i0 * st0 + i1 * st1

    # Perform the sum reduction over the specified dimension in chunks
    acc = 0.0
    for r_start in range(0, R, BLOCK_SIZE_R):
        offsets = r_start + tl.arange(0, BLOCK_SIZE_R)
        mask = offsets < R
        # Load the block of elements along the reduction dimension
        val = tl.load(x_ptr + base_off + offsets * r_stride, mask=mask, other=0.0)
        # Sum the block and add to accumulator
        acc += tl.sum(val, axis=0)

    # Store the final summed value in the output tensor
    tl.store(out_ptr + m_idx, acc)

def triton_sum_reduction(x: torch.Tensor, dim: int):
    """
    Wrapper for the Triton sum reduction kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    
    # Ensure input is contiguous to simplify stride calculations if necessary, 
    # though the kernel handles strides.
    S = x.shape
    S0, S1, S2 = S
    st0, st1, st2 = x.stride()
    
    R = S[dim]
    M = x.numel() // R
    
    # Prepare output shape: (..., 1, ...)
    out_shape = list(S)
    out_shape[dim] = 1
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    
    # Use a block size for the reduction dimension
    BLOCK_SIZE_R = 1024
    
    # Launch grid: one program per output element
    grid = (M,)
    
    sum_reduction_kernel[grid](
        x, out,
        S0, S1, S2,
        st0, st1, st2,
        dim,
        R,
        M,
        BLOCK_SIZE_R=BLOCK_SIZE_R
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction using a custom Triton kernel.
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
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        # Ensure we are working with FP32 as requested
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            
        return triton_sum_reduction(x, self.dim)