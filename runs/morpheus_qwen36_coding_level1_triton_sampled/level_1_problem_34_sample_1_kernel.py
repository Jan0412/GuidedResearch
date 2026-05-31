import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr, out_ptr, gamma_ptr, beta_ptr,
    N, C, H, W, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base_offset = n * C * H * W + c * H * W
    num_elements = H * W

    # Pass 1: Compute mean
    sum_x = 0.0
    for i in range(0, num_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        x = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
        sum_x += tl.sum(x)
    mean = sum_x / num_elements

    # Pass 2: Compute variance
    sum_var = 0.0
    for i in range(0, num_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        x = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
        sum_var += tl.sum((x - mean) ** 2)
    var = sum_var / num_elements
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 3: Normalize and apply affine parameters
    for i in range(0, num_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        x = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
        out = (x - mean) * rstd
        if gamma_ptr is not None:
            out = out * tl.load(gamma_ptr + c)
            out = out + tl.load(beta_ptr + c)
        tl.store(out_ptr + base_offset + offsets, out, mask=mask)


def triton_instance_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)

    N, C, H, W = x.shape
    BLOCK_SIZE = 256
    grid = (N * C,)

    instance_norm_kernel[grid](x, out, gamma, beta, N, C, H, W, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_instance_norm(x, self.gamma, self.beta, self.eps)