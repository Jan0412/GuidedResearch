import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    stride_b,
    stride_f,
    stride_d1,
    stride_d2,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
    FEATURES: tl.constexpr,
):
    # Each program handles one output element (b, d1, d2)
    idx = tl.program_id(0)
    num_elements = n_elements
    
    # Calculate coordinates from linear index
    b = idx // (FEATURES * stride_d1)
    d1_idx = (idx % (FEATURES * stride_d1)) // stride_d1
    d2_idx = (idx % stride_d1) // 1  # Assuming contiguous dim2
    
    # Pointer to the start of the feature dimension for this (b, d1, d2)
    base_ptr = x_ptr + b * stride_b + d1_idx * stride_d1 + d2_idx
    
    # Compute sum of squares along feature dimension
    sum_sq = 0.0
    for f in range(FEATURES):
        x_val = tl.load(base_ptr + f * stride_f, mask=True, other=0.0)
        sum_sq += x_val * x_val
    
    # Compute RMS
    rms = tl.sqrt(sum_sq / FEATURES + eps)
    
    # Normalize and store
    for f in range(FEATURES):
        x_val = tl.load(base_ptr + f * stride_f, mask=True, other=0.0)
        out_val = x_val / rms
        tl.store(out_ptr + f * stride_f, out_val, mask=True)


def triton_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Applies RMS Normalization using a custom Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, features, dim1, dim2).
        eps (float): Small value added to the denominator to avoid division by zero.
        
    Returns:
        torch.Tensor: Output tensor with RMS Normalization applied.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, features, dim1, dim2 = x.shape
    n_elements = batch_size * features * dim1 * dim2
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Determine grid size (one program per output element)
    grid = (n_elements,)
    
    # Launch the Triton kernel
    rms_norm_kernel[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        n_elements, eps,
        BLOCK_SIZE=features,
        FEATURES=features,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using a custom Triton kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer with Triton optimization.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_rms_norm(x, self.eps)