import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weight (optional, not used in this implementation but kept for interface)
    B,  # pointer to the bias (optional, not used in this implementation but kept for interface)
    stride_xm, stride_xn,  # strides for input tensor
    stride_ym, stride_yn,  # strides for output tensor
    M, N,  # dimensions
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get the program id
    pid_m = tl.program_id(0)
    
    # Create a range of offsets for rows (batch dimension)
    row_start = pid_m * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    mask_m = row_offsets < M
    
    # Load the input row by row
    # We'll compute RMS per row (across N features)
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        mask_n = n_offsets < N
        mask = mask_m[:, None] & mask_n[None, :]
        
        # Load input values
        x = tl.load(X + row_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn, mask=mask, other=0.0)
        
        # Compute sum of squares
        sq_sum = tl.sum(x * x, axis=1, keep_dims=True)
        
        # Compute mean
        mean_sq = sq_sum / N
        
        # Compute RMS = sqrt(mean_sq + eps)
        rms = tl.sqrt(mean_sq + eps)
        
        # Normalize and store
        y = x / rms
        
        # Store output
        tl.store(Y + row_offsets[:, None] * stride_ym + n_offsets[None, :] * stride_yn, y, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float = 1e-5):
    """
    Apply RMS Normalization using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, *)
        eps: Small value to avoid division by zero
    
    Returns:
        Output tensor with RMS normalization applied
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get original shape
    original_shape = x.shape
    batch_size = x.shape[0]
    num_features = x.shape[1]
    
    # Reshape to 2D: (batch_size, num_features * dim1 * dim2 * ...)
    x_2d = x.view(batch_size, -1)
    
    # Get dimensions
    M, N = x_2d.shape
    
    # Create output tensor
    y = torch.empty_like(x_2d)
    
    # Set block sizes for kernel
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    
    # Calculate grid size
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
    )
    
    # Launch kernel
    rms_norm_kernel[grid](
        x_2d, y, None, None,
        x_2d.stride(0), x_2d.stride(1),
        y.stride(0), y.stride(1),
        M, N,
        eps=eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Reshape back to original shape
    return y.view(original_shape)


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using Triton kernel.
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
        Applies RMS Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_rms_norm(x, self.eps)