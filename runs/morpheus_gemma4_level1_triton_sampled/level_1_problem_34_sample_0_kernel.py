import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def instnorm_kernel(
    x_ptr, 
    out_ptr, 
    n_elements_spatial, 
    num_features, 
    stride_n, 
    stride_c, 
    eps, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel of one instance
    pid = tl.program_id(0)
    n = pid // num_features
    c = pid % num_features

    # Pointer to the start of the spatial dimensions for this (n, c)
    x_slice_ptr = x_ptr + n * stride_n + c * stride_c
    out_slice_ptr = out_ptr + n * stride_n + c * stride_c

    # Pass 1: Compute mean and variance
    sum_x = 0.0
    sum_x2 = 0.0
    for i in range(0, n_elements_spatial, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_spatial
        x = tl.load(x_slice_ptr + offsets, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / n_elements_spatial
    # Variance = E[X^2] - (E[X])^2
    var = (sum_x2 / n_elements_spatial) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: Normalize and store
    for i in range(0, n_elements_spatial, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_spatial
        x = tl.load(x_slice_ptr + offsets, mask=mask, other=0.0)
        out = (x - mean) * inv_std
        tl.store(out_slice_ptr + offsets, out, mask=mask)

def triton_instance_norm(x: torch.Tensor, num_features: int, eps: float = 1e-5):
    """
    Triton wrapper for Instance Normalization.
    Input x shape: (N, C, H, W)
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    n, c, h, w = x.shape
    assert c == num_features, "Input channels must match num_features."

    out = torch.empty_like(x)
    n_elements_spatial = h * w
    
    # Strides for a contiguous (N, C, H, W) tensor
    stride_n = c * h * w
    stride_c = h * w

    BLOCK_SIZE = 1024
    # Grid: one program per instance per channel
    grid = (n * c,)

    instnorm_kernel[grid](
        x, out, 
        n_elements_spatial, 
        num_features, 
        stride_n, 
        stride_c, 
        eps, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using custom Triton kernels.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x, self.num_features)