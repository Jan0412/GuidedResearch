import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    N,
    C,
    H,
    W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample-channel pair
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    # Compute mean and variance using shared memory reduction
    # Initialize accumulators
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Load data in blocks and accumulate
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H * W
        # Load data for current sample and channel
        x = tl.load(x_ptr + n * stride_n + c * stride_c + offsets * stride_w, mask=mask, other=0.0)
        sum_val += x
        sum_sq_val += x * x

    # Reduce across blocks using shared memory
    # Use atomic operations for reduction
    # This is a simplified reduction; for production, use a more efficient shared memory reduction
    tl.atomic_add(mean_ptr + pid, sum_val.sum())
    tl.atomic_add(var_ptr + pid, sum_sq_val.sum())

    # Compute mean and variance
    mean = tl.load(mean_ptr + pid) / (H * W)
    var = tl.load(var_ptr + pid) / (H * W) - mean * mean
    std = tl.sqrt(var + eps)

    # Normalize and apply affine
    for i in range(0, H * W, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H * W
        x = tl.load(x_ptr + n * stride_n + c * stride_c + offsets * stride_w, mask=mask, other=0.0)
        x_norm = (x - mean) / std
        # Load weight and bias
        w = tl.load(weight_ptr + c)
        b = tl.load(bias_ptr + c)
        out = w * x_norm + b
        tl.store(out_ptr + n * stride_n + c * stride_c + offsets * stride_w, out, mask=mask)


def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C, H, W = x.shape
    out = torch.empty_like(x)

    # Allocate memory for mean and variance
    mean = torch.zeros((N, C), dtype=torch.float32, device=x.device)
    var = torch.zeros((N, C), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = 1024  # Tunable parameter for block size
    grid = (N * C,)

    # Launch the Triton kernel
    instance_norm_kernel[grid](
        x, out, weight, bias, mean, var,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        N, C, H, W, eps, BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_instance_norm(x, self.weight, self.bias, self.eps)