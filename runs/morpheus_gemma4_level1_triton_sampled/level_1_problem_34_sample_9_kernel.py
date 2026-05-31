import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def instance_norm_kernel(
    x_ptr, 
    out_ptr,
    n_spatial,
    stride_n, 
    stride_c,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one instance (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    # Pointer to the start of the current instance (n, c, 0, 0)
    # x shape is (N, C, H, W), n_spatial = H * W
    ptr = x_ptr + n * stride_n + c * stride_c
    out_ptr_instance = out_ptr + n * stride_n + c * stride_c
    
    # First pass: Compute sum and sum of squares for mean and variance
    sum_val = 0.0
    sum_sq_val = 0.0
    
    for i in range(0, n_spatial, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_spatial
        vals = tl.load(ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        sum_sq_val += tl.sum(vals * vals, axis=0)
        
    mean = sum_val / n_spatial
    # Variance = E[X^2] - (E[X])^2
    var = (sum_sq_val / n_spatial) - (mean * mean)
    # Use tl.maximum to prevent negative variance due to precision
    inv_std = 1.0 / tl.sqrt(tl.maximum(var, 0.0) + 1e-5)
    
    # Second pass: Normalize and store the result
    for i in range(0, n_spatial, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_spatial
        vals = tl.load(ptr + offsets, mask=mask, other=0.0)
        res = (vals - mean) * inv_std
        tl.store(out_ptr_instance + offsets, res, mask=mask)

def triton_instance_norm(x: torch.Tensor):
    """
    Triton wrapper for Instance Normalization.
    Assumes input x is (N, C, H, W).
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    N, C, H, W = x.shape
    n_spatial = H * W
    
    out = torch.empty_like(x)
    
    # Strides for (N, C, H, W)
    stride_n = C * n_spatial
    stride_c = n_spatial
    
    # Grid: one program per (N, C) pair
    grid = (N * C,)
    
    # Launch kernel
    instance_norm_kernel[grid](
        x, out, 
        n_spatial, 
        stride_n, stride_c, 
        C, 
        BLOCK_SIZE=1024
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using a custom Triton kernel.
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
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x)