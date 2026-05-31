import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def instance_norm_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    stride_n, 
    stride_c,
    num_features,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID to determine which (batch, channel) pair this program handles
    pid = tl.program_id(0)
    n = pid // num_features
    c = pid % num_features

    # Calculate the offset to the start of the specific instance's channel
    # x shape: (N, C, H, W)
    offset_nc = n * stride_n + c * stride_c
    
    # Pointers for the current instance-channel slice
    curr_x_ptr = x_ptr + offset_nc
    curr_out_ptr = out_ptr + offset_nc

    # Pass 1: Compute Mean and Variance
    # We use the formula: var = E[x^2] - (E[x])^2
    sum_x = 0.0
    sum_x2 = 0.0
    
    i = 0
    while i < n_elements:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        val = tl.load(curr_x_ptr + offsets, mask=mask, other=0.0)
        
        sum_x += tl.sum(val, axis=0)
        sum_x2 += tl.sum(val * val, axis=0)
        i += BLOCK_SIZE

    mean = sum_x / n_elements
    var = (sum_x2 / n_elements) - (mean * mean)
    # Ensure variance is non-negative due to potential floating point precision issues
    var = tl.maximum(var, 0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: Normalize and Store
    i = 0
    while i < n_elements:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        val = tl.load(curr_x_ptr + offsets, mask=mask, other=0.0)
        
        out = (val - mean) * inv_std
        tl.store(curr_out_ptr + offsets, out, mask=mask)
        i += BLOCK_SIZE

def triton_instance_norm(x: torch.Tensor, num_features: int, eps: float = 1e-5):
    """
    Wrapper for the Triton Instance Normalization kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    N, C, H, W = x.shape
    n_elements = H * W
    out = torch.empty_like(x)

    # Strides for indexing (N, C, H, W)
    stride_n = C * H * W
    stride_c = H * W

    # One program per (batch_size * num_features)
    grid = (N * C,)
    
    # BLOCK_SIZE is a tunable parameter. 1024 is generally a good starting point for FP32.
    BLOCK_SIZE = 1024
    
    instance_norm_kernel[grid](
        x, 
        out, 
        n_elements, 
        stride_n, 
        stride_c, 
        num_features, 
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
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x, self.num_features, self.eps)