import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, 
    out_ptr, 
    s0, s1, s2, s3, 
    B, F, D1, D2, 
    eps, 
    BLOCK_F: tl.constexpr, 
    BLOCK_D2: tl.constexpr,
):
    # Grid dimensions: (B, D1, (D2 + BLOCK_D2 - 1) // BLOCK_D2)
    pid_b = tl.program_id(0)
    pid_d1 = tl.program_id(1)
    pid_d2_block = tl.program_id(2)

    # Base pointer for the current block in the D2 dimension
    # x[pid_b, :, pid_d1, pid_d2_block * BLOCK_D2]
    base_ptr = pid_b * s0 + pid_d1 * s2 + (pid_d2_block * BLOCK_D2) * s3

    # Create offsets for the feature dimension (F) and the block of the last dimension (D2)
    off_f = tl.arange(0, BLOCK_F)
    off_d2 = tl.arange(0, BLOCK_D2)

    # Create masks to handle boundaries
    mask_f = off_f < F
    mask_d2 = (pid_d2_block * BLOCK_D2 + off_d2) < D2
    mask = mask_f[:, None] & mask_d2[None, :]

    # Load the input block of shape (BLOCK_F, BLOCK_D2)
    # x[pid_b, off_f, pid_d1, pid_d2_block * BLOCK_D2 + off_d2]
    x = tl.load(x_ptr + base_ptr + off_f[:, None] * s1 + off_d2[None, :] * s3, mask=mask, other=0.0)

    # Calculate the sum of squares along the feature dimension (axis=0)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    
    # RMS = sqrt(mean(x^2) + eps)
    mean_sq = sum_sq / F
    rms = tl.sqrt(mean_sq + eps)

    # Normalize the input by dividing by the RMS
    # rms has shape (BLOCK_D2,), we broadcast it to (BLOCK_F, BLOCK_D2)
    out = x / rms[None, :]

    # Store the result back to memory
    tl.store(out_ptr + base_ptr + off_f[:, None] * s1 + off_d2[None, :] * s3, out, mask=mask)


def triton_rmsnorm(x: torch.Tensor, eps: float):
    """
    Triton wrapper for the RMS Normalization kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    
    # Ensure the tensor is contiguous to make stride calculations straightforward
    x = x.contiguous()
    B, F, D1, D2 = x.shape
    s0, s1, s2, s3 = x.stride()
    
    out = torch.empty_like(x)
    
    # Block sizes: F is usually small enough to fit in SRAM. 
    # We use the next power of 2 for Triton's requirements.
    BLOCK_F = triton.next_power_of_2(F)
    BLOCK_D2 = 32 # Tunable parameter for the last dimension block size

    # Grid: one program for each (batch, dim1) and a block of dim2
    grid = (B, D1, (D2 + BLOCK_D2 - 1) // BLOCK_D2)

    rms_norm_kernel[grid](
        x, out, 
        s0, s1, s2, s3, 
        B, F, D1, D2, 
        eps, 
        BLOCK_F=BLOCK_F, 
        BLOCK_D2=BLOCK_D2
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
        Applies RMS Normalization to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_rmsnorm(x, self.eps)