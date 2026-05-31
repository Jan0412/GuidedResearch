import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, H, W, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    x_ptr += n * C * H * W + c * H * W
    y_ptr += n * C * H * W + c * H * W

    num_elements = H * W
    sum_val = 0.0
    sum_sq_val = 0.0

    # Pass 1: Compute sum and sum of squares over spatial dimensions
    for start in range(0, num_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x, mask=mask, other=0.0)
        sum_sq_val += tl.sum(x * x, mask=mask, other=0.0)

    mean = sum_val / num_elements
    var = sum_sq_val / num_elements - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + c)
    b = tl.load(bias_ptr + c)

    # Pass 2: Normalize and apply affine transformation
    for start in range(0, num_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = (x - mean) * inv_std * w + b
        tl.store(y_ptr + offsets, y, mask=mask)


def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)

    N, C, H, W = x.shape
    BLOCK_SIZE = 1024  # Tunable block size for spatial dimensions

    grid = (N * C,)
    instance_norm_kernel[grid](
        x, out, weight, bias,
        N, C, H, W, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features, affine=True, track_running_stats=False)
        self.eps = self.inorm.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_instance_norm(x, self.inorm.weight, self.inorm.bias, self.eps)


def get_inputs():
    x = torch.rand(112, 64, 512, 512).cuda()
    return [x]

def get_init_inputs():
    return [64]