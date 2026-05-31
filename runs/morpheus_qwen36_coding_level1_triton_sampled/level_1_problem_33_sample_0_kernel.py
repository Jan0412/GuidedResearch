import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batchnorm2d_kernel(
    x_ptr, out_ptr,
    weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    C, H, W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < C * H * W

    # Load input tensor elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index for each element in the flattened view
    HW = H * W
    c = (offsets // HW) % C

    # Load per-channel parameters
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    mu = tl.load(running_mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + c, mask=mask, other=1.0)

    # Compute normalized output: (x - mu) / sqrt(var + eps) * w + b
    inv_std = tl.rsqrt(var + eps)
    out = (x - mu) * inv_std * w + b

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_batchnorm2d(x, weight, bias, running_mean, running_var, eps=1e-5):
    """Wrapper to launch the custom BatchNorm2d Triton kernel."""
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    
    # Ensure contiguous memory layout for optimal coalesced access
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()

    out = torch.empty_like(x)
    C, H, W = x.shape[1], x.shape[2], x.shape[3]
    num_elements = C * H * W
    
    # Tunable block size for FP32 throughput optimization
    BLOCK_SIZE = 256

    # Compute grid size
    grid = ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch kernel
    batchnorm2d_kernel[grid](
        x, out,
        weight, bias, running_mean, running_var,
        C, H, W,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model replacing nn.BatchNorm2d with a custom FP32 Triton kernel.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_batchnorm2d(x, self.weight, self.bias, self.running_mean, self.running_var, self.eps)