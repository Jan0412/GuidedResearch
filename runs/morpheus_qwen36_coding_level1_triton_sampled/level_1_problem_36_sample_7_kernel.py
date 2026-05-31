import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    features,
    dim1,
    dim2,
    eps,
    BLOCK_SIZE_F: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_f = tl.arange(0, BLOCK_SIZE_F)
    mask_f = offsets_f < features
    offsets = pid + offsets_f * dim1 * dim2
    
    x = tl.load(x_ptr + offsets, mask=mask_f, other=0.0)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    
    rms = tl.sqrt(sum_sq / features + eps)
    out = x / rms
    
    tl.store(out_ptr + offsets, out, mask=mask_f)


def triton_rms_norm(x: torch.Tensor, eps: float, features: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    
    batch_size = x.shape[0]
    dim1 = x.shape[2]
    dim2 = x.shape[3]
    num_groups = batch_size * dim1 * dim2
    
    BLOCK_SIZE_F = 128
    
    grid = (num_groups,)
    
    rms_norm_kernel[grid](
        x, out, features, dim1, dim2, eps, BLOCK_SIZE_F=BLOCK_SIZE_F
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(x, self.eps, self.num_features)