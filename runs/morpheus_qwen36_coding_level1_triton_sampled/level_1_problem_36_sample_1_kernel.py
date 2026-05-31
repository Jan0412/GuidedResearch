import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    stride_x,
    features,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Each program handles one normalization vector of length 'features'
    # Vectors are strided by stride_x in the contiguous memory layout
    offsets = pid * stride_x + tl.arange(0, BLOCK_SIZE) * stride_x
    mask = tl.arange(0, BLOCK_SIZE) < features

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute RMS inverse: 1 / sqrt(mean(x^2) + eps)
    x2 = x * x
    sum_x2 = tl.sum(x2, axis=0)
    rms_inv = tl.rsqrt(sum_x2 / features + eps)

    # Apply normalization
    out = x * rms_inv

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    batch_size, features, dim1, dim2 = x.shape
    stride_x = x.stride(1)  # Stride for the feature dimension

    # Use a block size that is a power of 2 and covers the feature dimension
    BLOCK_SIZE = triton.next_power_of_2(features)

    # Number of independent normalization vectors
    num_norms = batch_size * dim1 * dim2
    grid = (num_norms,)

    rms_norm_kernel[grid](
        x, out,
        stride_x, features, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(x, self.eps)


def get_inputs():
    batch_size = 112
    features = 64
    dim1 = 512
    dim2 = 512
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    features = 64
    return [features]