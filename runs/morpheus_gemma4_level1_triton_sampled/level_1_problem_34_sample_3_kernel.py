import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def instance_norm_kernel(
    x_ptr, 
    out_ptr, 
    stride_b, 
    stride_c, 
    n_elements, 
    eps, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    
    # Calculate batch and channel indices
    # We assume the grid is (batch_size * num_features,)
    # To correctly decode pid, we need the number of features. 
    # However, it's easier to pass the total offset or calculate it outside.
    # For simplicity, we can pass the base pointer for each (b, c) 
    # or calculate it using the strides.
    
    # In this implementation, we assume the grid is mapped such that 
    # pid = b * num_features + c. 
    # Since we don't have num_features as a constexpr, we calculate the 
    # pointer offset based on the pid and the stride_c (which is H*W).
    
    # The starting pointer for the current (batch, channel) instance
    # Note: pid is the index of the (b, c) pair.
    # offset = (b * stride_b) + (c * stride_c)
    # Since we know stride_c is the size of one instance, 
    # and pid is the instance index:
    instance_ptr = x_ptr + pid * stride_c
    out_instance_ptr = out_ptr + pid * stride_c

    # First pass: Calculate mean and variance
    sum_val = 0.0
    sum_sq_val = 0.0
    
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(instance_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq_val += tl.sum(x * x, axis=0)
    
    mean = sum_val / n_elements
    # Population variance: E[X^2] - (E[X])^2
    var = (sum_sq_val / n_elements) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: Normalize and store
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(instance_ptr + offsets, mask=mask, other=0.0)
        out = (x - mean) * inv_std
        tl.store(out_instance_ptr + offsets, out, mask=mask)

def triton_instance_norm(x: torch.Tensor, eps: float = 1e-5):
    """
    Triton wrapper for Instance Normalization.
    x: (B, C, H, W)
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure input is contiguous to simplify pointer arithmetic
    x = x.contiguous()
    B, C, H, W = x.shape
    n_elements = H * W
    
    out = torch.empty_like(x)
    
    # Strides for the input tensor
    stride_b = C * H * W
    stride_c = H * W
    
    # Grid: one program per (batch, channel) pair
    grid = (B * C,)
    BLOCK_SIZE = 1024
    
    instance_norm_kernel[grid](
        x, out, 
        stride_b, stride_c, 
        n_elements, eps, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using Triton kernels.
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
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x, eps=self.eps)