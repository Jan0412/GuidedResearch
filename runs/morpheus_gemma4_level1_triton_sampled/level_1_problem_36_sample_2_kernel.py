import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr,
    out_ptr,
    eps,
    n_features,
    S0, S1, S2, S3,
    B, D1, D2,
    BLOCK_SIZE_V: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr,
):
    # Each program processes a block of vectors
    pid = tl.program_id(0)
    
    # Vector indices for the current block
    v_idx = pid * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    mask_v = v_idx < (B * D1 * D2)
    
    # Decompose flat vector index into (b, d1, d2)
    # b = v_idx // (D1 * D2)
    # rem = v_idx % (D1 * D2)
    # d1 = rem // D2
    # d2 = rem % D2
    
    # Optimization: Calculate base pointers for the vectors in the block
    # base_ptr = x_ptr + b * S0 + d1 * S2 + d2 * S3
    # Since S2=D2 and S3=1 for contiguous tensors, b*S0 + rem is often sufficient,
    # but we use the general formula for robustness.
    
    b = v_idx // (D1 * D2)
    rem = v_idx % (D1 * D2)
    d1 = rem // D2
    d2 = rem % D2
    
    base_ptrs = x_ptr + b * S0 + d1 * S2 + d2 * S3
    
    # Feature indices for the reduction
    f_idx = tl.arange(0, BLOCK_SIZE_F)
    mask_f = f_idx < n_features
    
    # Create 2D pointers: [BLOCK_SIZE_V, BLOCK_SIZE_F]
    # ptrs[v, f] = base_ptrs[v] + f * S1
    ptrs = base_ptrs[:, None] + f_idx[None, :] * S1
    mask = mask_v[:, None] & mask_f[None, :]
    
    # Load data
    x = tl.load(ptrs, mask=mask, other=0.0)
    
    # Calculate RMS: sqrt(mean(x^2) + eps)
    x_sq = x * x
    # Sum across the feature dimension (axis 1)
    sum_sq = tl.sum(x_sq, axis=1)
    mean_sq = sum_sq / n_features
    rms = tl.sqrt(mean_sq + eps)
    
    # Normalize
    out = x / rms[:, None]
    
    # Store result
    tl.store(ptrs, out, mask=mask)

def triton_rmsnorm(x: torch.Tensor, eps: float):
    # Ensure input is on CUDA and contiguous
    assert x.is_cuda, "Tensors must be on CUDA."
    
    B, F, D1, D2 = x.shape
    S0, S1, S2, S3 = x.stride()
    
    out = torch.empty_like(x)
    
    # Tuning parameters
    BLOCK_SIZE_V = 16
    BLOCK_SIZE_F = 128 # Must be >= F
    
    n_vectors = B * D1 * D2
    grid = ((n_vectors + BLOCK_SIZE_V - 1) // BLOCK_SIZE_V,)
    
    rmsnorm_kernel[grid](
        x, out, 
        eps, F, 
        S0, S1, S2, S3, 
        B, D1, D2, 
        BLOCK_SIZE_V=BLOCK_SIZE_V, 
        BLOCK_SIZE_F=BLOCK_SIZE_F
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using Triton kernels.
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
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_rmsnorm(x, self.eps)