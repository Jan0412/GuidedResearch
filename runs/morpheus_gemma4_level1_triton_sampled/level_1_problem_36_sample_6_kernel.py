import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr,
    out_ptr,
    stride_b,
    stride_f,
    stride_d1,
    stride_d2,
    B,
    F,
    D1,
    D2,
    eps,
    BLOCK_F: tl.constexpr,
    BLOCK_D2: tl.constexpr,
):
    # Each program handles a block of D2 elements for a specific (batch, d1)
    pid_bd1 = tl.program_id(0)
    pid_d2 = tl.program_id(1)

    # Decompose pid_bd1 into batch (b) and d1
    b = pid_bd1 // D1
    d1 = pid_bd1 % D1
    d2_start = pid_d2 * BLOCK_D2

    # Create offsets for the feature dimension (F) and the D2 dimension
    f_offsets = tl.arange(0, BLOCK_F)
    d2_offsets = d2_start + tl.arange(0, BLOCK_D2)

    # Masks to avoid out-of-bounds access
    mask_f = f_offsets < F
    mask_d2 = d2_offsets < D2
    mask = mask_f[:, None] & mask_d2[None, :]

    # Calculate the pointer for the 2D block [BLOCK_F, BLOCK_D2]
    # x shape: (B, F, D1, D2)
    # offset = b * stride_b + f * stride_f + d1 * stride_d1 + d2 * stride_d2
    ptr = x_ptr + b * stride_b + f_offsets[:, None] * stride_f + d1 * stride_d1 + d2_offsets[None, :] * stride_d2

    # Load the block of data
    x = tl.load(ptr, mask=mask, other=0.0)

    # Calculate the sum of squares along the feature dimension (axis=0)
    # We use float32 for the accumulation to maintain precision
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)

    # RMS = sqrt(mean(x^2) + eps)
    rms = tl.sqrt(sum_sq / F + eps)

    # Normalize: x / rms
    # rms has shape (BLOCK_D2,), x has shape (BLOCK_F, BLOCK_D2)
    out = x / rms[None, :]

    # Store the result back to memory
    tl.store(ptr, out, mask=mask)

def triton_rmsnorm(x: torch.Tensor, eps: float):
    # Ensure input is contiguous for predictable strides, though we pass strides explicitly
    # The input shape is (B, F, D1, D2)
    B, F, D1, D2 = x.shape
    stride_b, stride_f, stride_d1, stride_d2 = x.stride()
    
    out = torch.empty_like(x)
    
    # Tuning parameters
    # Since F is typically small (e.g., 64), we can fit it in a single block.
    # We choose BLOCK_F to be a power of 2 that covers F.
    BLOCK_F = triton.next_power_of_2(F)
    BLOCK_D2 = 32 # Standard block size for memory coalescing

    # Grid: one program per (batch, d1) and one program per block of D2
    grid = (B * D1, (D2 + BLOCK_D2 - 1) // BLOCK_D2)

    rmsnorm_kernel[grid](
        x, out,
        stride_b, stride_f, stride_d1, stride_d2,
        B, F, D1, D2,
        eps,
        BLOCK_F=BLOCK_F,
        BLOCK_D2=BLOCK_D2,
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
        Applies RMS Normalization to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # The Triton kernel expects the tensor to be on GPU
        if not x.is_cuda:
            # Fallback to PyTorch for CPU tensors
            rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
            return x / rms
            
        return triton_rmsnorm(x, self.eps)