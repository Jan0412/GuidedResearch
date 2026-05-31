import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, out_ptr,
    stride_b, stride_f, stride_d1, stride_d2,
    n_features, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for RMS Normalization.
    Each program processes one normalization vector along the feature dimension (dim=1).
    """
    # Grid: (batch_size, dim1, dim2)
    b = tl.program_id(0)
    d1 = tl.program_id(1)
    d2 = tl.program_id(2)

    # Calculate the base pointer for the feature vector at coordinates (b, d1, d2)
    # Input shape: (batch_size, num_features, dim1, dim2)
    base_ptr = x_ptr + b * stride_b + d1 * stride_d1 + d2 * stride_d2
    
    # Create offsets for the feature dimension
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_features
    
    # Load the feature vector from memory
    x = tl.load(base_ptr + offsets * stride_f, mask=mask)
    
    # Compute the Root Mean Square (RMS)
    # Formula: rms = sqrt(mean(x^2) + eps)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / n_features
    rms = tl.sqrt(mean_sq + eps)
    
    # Normalize the vector: divide each element by the RMS
    out = x / rms
    
    # Store the normalized result back to the output tensor
    out_ptr_base = out_ptr + b * stride_b + d1 * stride_d1 + d2 * stride_d2
    tl.store(out_ptr_base + offsets * stride_f, out, mask=mask)

def triton_rmsnorm(x: torch.Tensor, eps: float):
    """
    Wrapper for the Triton RMSNorm kernel.
    """
    # Ensure the tensor is in FP32 for precision and on GPU
    x = x.to(torch.float32)
    
    B, F, D1, D2 = x.shape
    
    # Ensure tensors are contiguous to ensure stride logic is consistent
    x = x.contiguous()
    out = torch.empty_like(x)
    
    # Extract strides for the (B, F, D1, D2) layout
    stride_b, stride_f, stride_d1, stride_d2 = x.stride()
    
    # BLOCK_SIZE must be a power of 2 for Triton efficiency
    BLOCK_SIZE = triton.next_power_of_2(F)
    
    # Launch a grid of programs, one for each normalization group
    grid = (B, D1, D2)
    
    rms_norm_kernel[grid](
        x, out,
        stride_b, stride_f, stride_d1, stride_d2,
        F, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using a custom Triton kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Replace the PyTorch implementation with the Triton-optimized version
        return triton_rmsnorm(x, self.eps)