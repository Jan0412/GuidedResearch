import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bn_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    C,
    H,
    W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate offsets for the current block
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index for broadcasting
    # Shape is [N, C, H, W], stride for C is H * W
    stride_c = H * W
    c_idx = (offsets // stride_c) % C

    # Load parameters and statistics using channel indices
    # These are broadcasted to the batch/spatial dimensions via masking
    running_mean = tl.load(running_mean_ptr + c_idx, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + c_idx, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + c_idx, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # Batch Normalization formula: y = (x - mean) / sqrt(var + eps) * weight + bias
    eps = 1e-5
    out = (x - running_mean) * tl.rsqrt(running_var + eps) * weight + bias

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    """
    Wrapper function to launch the Triton Batch Normalization kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]

    # Tunable block size
    BLOCK_SIZE = 128

    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    bn_kernel[grid](
        x, out, weight, bias, running_mean, running_var,
        C, H, W, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for Batch Normalization.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the BatchNorm layer
        weight = self.bn.weight
        bias = self.bn.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        eps = self.bn.eps

        # Use custom Triton kernel for inference
        return triton_bn(x, weight, bias, running_mean, running_var, eps)