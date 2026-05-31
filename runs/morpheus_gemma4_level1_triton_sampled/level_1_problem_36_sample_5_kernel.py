import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, 
    out_ptr, 
    N, 
    eps, 
    stride_b, 
    stride_n, 
    stride_d1, 
    stride_d2,
    B, 
    D1, 
    D2,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for RMS Normalization.
    Each program handles one vector of size N along the normalization dimension.
    """
    pid = tl.program_id(0)
    
    # Decompose pid into coordinates (b, d1, d2)
    d2 = pid % D2
    d1 = (pid // D2) % D1
    b = pid // (D1 * D2)
    
    # Calculate the base pointer for the specific vector to be normalized.
    # The vector is located at [b, :, d1, d2]
    offset = b * stride_b + d1 * stride_d1 + d2 * stride_d2
    
    # Create offsets for the normalization dimension (N)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load the vector from memory
    # The normalization dimension has a stride of stride_n
    x = tl.load(x_ptr + offset + offsets * stride_n, mask=mask, other=0.0)
    
    # Compute the Root Mean Square (RMS)
    # rms = sqrt(mean(x^2) + eps)
    sq_sum = tl.sum(x * x, axis=0)
    rms = tl.sqrt(sq_sum / N + eps)
    
    # Normalize the vector
    out = x / rms
    
    # Store the result back to memory
    tl.store(out_ptr + offset + offsets * stride_n, out, mask=mask)


def triton_rmsnorm(x: torch.Tensor, eps: float):
    """
    Wrapper function to launch the Triton RMSNorm kernel.
    """
    # Ensure tensor is on GPU and contiguous for simplicity, 
    # though the kernel handles strides.
    assert x.is_cuda, "Tensors must be on CUDA."
    
    B, N, D1, D2 = x.shape
    stride_b, stride_n, stride_d1, stride_d2 = x.stride()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Grid: One program per vector to be normalized
    grid = (B * D1 * D2,)
    
    # BLOCK_SIZE must be a power of 2 and >= N
    BLOCK_SIZE = 1 << (N - 1).bit_length()
    
    # Launch kernel
    rms_norm_kernel[grid](
        x, 
        out, 
        N, 
        eps, 
        stride_b, 
        stride_n, 
        stride_d1, 
        stride_d2,
        B, 
        D1, 
        D2,
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
        # Use the Triton-optimized RMSNorm implementation
        return triton_rmsnorm(x, self.eps)