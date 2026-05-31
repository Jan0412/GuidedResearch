import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def instance_norm_kernel(
    x_ptr, 
    out_ptr, 
    stride_n, 
    stride_c, 
    stride_h, 
    stride_w, 
    H, 
    W, 
    eps, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one instance (batch_idx, channel_idx)
    pid = tl.program_id(0)
    num_features = stride_n // stride_c
    
    batch_idx = pid // num_features
    channel_idx = pid % num_features

    # Compute the pointer to the start of the current instance
    instance_ptr = x_ptr + batch_idx * stride_n + channel_idx * stride_c
    out_instance_ptr = out_ptr + batch_idx * stride_n + channel_idx * stride_c

    n_elements_spatial = H * W
    
    # First pass: Compute mean and variance
    sum_val = 0.0
    sum_sq_val = 0.0
    
    for i in range(0, n_elements_spatial, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_spatial
        # Flatten spatial dimensions for easier access
        # x is (N, C, H, W), we treat H*W as a flat array per instance
        # Note: Since we are at the instance level, we can just offset from instance_ptr
        # as long as the spatial layout is contiguous.
        vals = tl.load(instance_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        sum_sq_val += tl.sum(vals * vals, axis=0)

    mean = sum_val / n_elements_spatial
    var = (sum_sq_val / n_elements_spatial) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: Normalize and store
    for i in range(0, n_elements_spatial, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_spatial
        vals = tl.load(instance_ptr + offsets, mask=mask, other=0.0)
        norm_vals = (vals - mean) * inv_std
        tl.store(out_instance_ptr + offsets, norm_vals, mask=mask)

def triton_instance_norm(x: torch.Tensor, eps: float = 1e-5):
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure the tensor is contiguous to simplify pointer arithmetic
    x = x.contiguous()
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    
    stride_n, stride_c, stride_h, stride_w = x.stride()
    
    # Each program handles one (batch, channel) pair
    grid = (N * C,)
    BLOCK_SIZE = 1024 # Tunable block size for spatial reduction
    
    instance_norm_kernel[grid](
        x, out, 
        stride_n, stride_c, stride_h, stride_w, 
        H, W, eps, 
        BLOCK_SIZE=BLOCK_SIZE
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
        # nn.InstanceNorm2d defaults to affine=False, track_running_stats=False
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x, eps=self.eps)