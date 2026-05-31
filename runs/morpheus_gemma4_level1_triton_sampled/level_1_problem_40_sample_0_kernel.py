import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,          # Pointer to input tensor
    weight_ptr,     # Pointer to gamma (weight)
    bias_ptr,       # Pointer to beta (bias)
    out_ptr,        # Pointer to output tensor
    stride_x_row,   # Stride between rows in the input/output tensors
    N,              # Number of elements in the normalization group
    eps,            # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one normalization group (one row)
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * stride_x_row
    out_row_ptr = out_ptr + row_idx * stride_x_row

    # First pass: Compute mean and variance
    # We use the formula Var = E[x^2] - (E[x])^2 for efficiency in a single pass
    sum_val = 0.0
    sum_sq_val = 0.0
    
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq_val += tl.sum(x * x, axis=0)
    
    mean = sum_val / N
    var = (sum_sq_val / N) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: Normalize, scale, and shift
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize and apply affine transformation
        out = (x - mean) * inv_std * w + b
        tl.store(out_row_ptr + offsets, out, mask=mask)

def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    """
    Triton wrapper for Layer Normalization.
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    
    # Keep track of original shape to restore it at the end
    orig_shape = x.shape
    
    # Flatten the normalization dimensions
    # weight and bias shape define the normalization group size N
    N = weight.numel()
    B = x.numel() // N
    
    # Ensure tensors are contiguous and viewed as (B, N)
    x = x.view(B, N).contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    out = torch.empty_like(x)
    
    # Stride between rows is N elements
    stride_x_row = N
    
    # Grid is one block per row
    grid = (B,)
    
    # Launch kernel
    layer_norm_kernel[grid](
        x, weight, bias, out,
        stride_x_row,
        N,
        eps,
        BLOCK_SIZE=1024
    )
    
    return out.view(orig_shape)

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
        # We keep the nn.LayerNorm to manage parameters (weight, bias) and eps
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied.
        """
        return triton_layer_norm(x, self.ln.weight, self.ln.bias, self.ln.eps)