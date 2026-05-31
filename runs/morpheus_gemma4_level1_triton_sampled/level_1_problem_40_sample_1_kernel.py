import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(
    X, Y, W, B,
    stride_x_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one instance of the normalized dimensions)
    row_idx = tl.program_id(0)
    X_row_ptr = X + row_idx * stride_x_row
    Y_row_ptr = Y + row_idx * stride_x_row

    # First pass: Compute mean and variance
    sum_x = 0.0
    sum_x2 = 0.0
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x_vals = tl.load(X_row_ptr + offsets, mask=mask, other=0.0)
        sum_x += tl.sum(x_vals, axis=0)
        sum_x2 += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_x / N
    var = (sum_x2 / N) - (mean * mean)
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: Normalize, scale, and shift
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x_vals = tl.load(X_row_ptr + offsets, mask=mask, other=0.0)
        w_vals = tl.load(W + offsets, mask=mask, other=0.0)
        b_vals = tl.load(B + offsets, mask=mask, other=0.0)
        
        y_vals = (x_vals - mean) * rstd * w_vals + b_vals
        tl.store(Y_row_ptr + offsets, y_vals, mask=mask)

def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    # Input x shape: (B, N)
    B, N = x.shape
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    out = torch.empty_like(x)
    
    # Tunable block size
    BLOCK_SIZE = 1024
    
    # Grid is one program per row
    grid = (B,)
    
    layernorm_kernel[grid](
        x, out, weight, bias,
        x.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using a custom Triton kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = torch.tensor(normalized_shape)
        # Calculate the total number of elements in the normalized dimensions
        self.N = torch.prod(self.normalized_shape).item()
        
        # Learnable parameters gamma and beta
        self.weight = nn.Parameter(torch.ones(self.N))
        self.bias = nn.Parameter(torch.zeros(self.N))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied.
        """
        orig_shape = x.shape
        # Flatten the leading dimensions and the normalized dimensions
        # x: (*, normalized_shape) -> (B, N)
        x_flat = x.view(-1, self.N)
        
        # Apply the Triton-accelerated LayerNorm
        out_flat = triton_layernorm(x_flat, self.weight, self.bias, self.eps)
        
        # Reshape back to original input shape
        return out_flat.view(orig_shape)