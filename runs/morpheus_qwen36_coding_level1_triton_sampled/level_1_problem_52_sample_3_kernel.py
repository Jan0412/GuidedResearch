import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    B: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    b = tl.program_id(0)
    j_block = tl.program_id(1)
    
    j_start = j_block * BLOCK_J
    j_end = tl.minimum(j_start + BLOCK_J, D2)
    j_offsets = j_start + tl.arange(0, BLOCK_J)
    mask_j = j_offsets < j_end
    
    min_val = tl.full((BLOCK_J,), float('inf'), dtype=tl.float32)
    min_idx = tl.zeros((BLOCK_J,), dtype=tl.int32)
    
    num_i_blocks = (D1 + BLOCK_I - 1) // BLOCK_I
    
    for i_block in range(num_i_blocks):
        i_start = i_block * BLOCK_I
        i_end = tl.minimum(i_start + BLOCK_I, D1)
        i_offsets = i_start + tl.arange(0, BLOCK_I)
        mask_i = i_offsets < i_end
        
        base_offset = b * D1 * D2 + i_start * D2 + j_start
        x_block = tl.load(
            x_ptr + base_offset,
            shape=(BLOCK_I, BLOCK_J),
            strides=(D2, 1),
            mask=mask_i[:, None] & mask_j[None, :],
            other=float('inf')
        )
        
        min_val_block = tl.min(x_block, axis=0)
        argmin_block = tl.argmin(x_block, axis=0)
        
        mask_update = min_val_block < min_val
        min_val = tl.where(mask_update, min_val_block, min_val)
        min_idx = tl.where(mask_update, i_start + argmin_block, min_idx)
        
    tl.store(out_ptr + b * D2 + j_offsets, min_idx, mask=mask_j)


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x.dtype == torch.float32, "Input tensor must be FP32."
    assert dim == 1, "Only dim=1 is supported for this kernel."
    
    x = x.contiguous()
    B, D1, D2 = x.shape
    
    out = torch.empty((B, D2), dtype=torch.int32, device=x.device)
    
    BLOCK_J = 32
    BLOCK_I = 64
    
    num_j_blocks = (D2 + BLOCK_J - 1) // BLOCK_J
    grid = (B, num_j_blocks)
    
    argmin_kernel[grid](
        x, out, B, D1, D2,
        BLOCK_J=BLOCK_J,
        BLOCK_I=BLOCK_I
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, self.dim)