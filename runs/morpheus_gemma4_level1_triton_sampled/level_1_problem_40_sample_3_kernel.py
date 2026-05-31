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
    stride_x_row,   # Stride between rows of x
    n_cols,         # Number of elements in the normalization dimension
    eps,            # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    
    # Offset for the current row
    row_start_ptr = x_ptr + row_idx * stride_x_row
    out_row_start_ptr = out_ptr + row_idx * stride_x_row

    # Load the row data
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)

    # Calculate mean
    mean = tl.sum(x * mask) / n_cols
    
    # Calculate variance
    # Var = E[x^2] - (E[x])^2
    # Using (x - mean)^2 for better numerical stability in FP32
    diff = x - mean
    var = tl.sum((diff * diff) * mask) / n_cols
    
    # Compute inverse standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Normalize and apply scale/shift
    out = diff * rstd * weight + bias
    
    # Store the result
    tl.store(out_row_start_ptr + offsets, out, mask=mask)


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton wrapper for Layer Normalization.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure inputs are contiguous for the kernel
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Output tensor
    out = torch.empty_like(x)
    
    # The normalization is performed over the last dimension
    n_cols = x.shape[-1]
    # Number of rows to process (all dimensions except the last)
    n_rows = x.numel() // n_cols
    # Stride to move to the next row
    stride_x_row = x.stride(-1) * n_cols if x.dim() > 1 else 1 # simplified for contiguous
    # For contiguous tensor, stride_x_row is simply n_cols
    stride_x_row = n_cols 

    # BLOCK_SIZE must be a power of 2 and at least as large as n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Grid is one program per row
    grid = (n_rows,)
    
    layer_norm_kernel[grid](
        x, weight, bias, out,
        stride_x_row,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
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
        # Store normalized shape to handle input validation if necessary
        self.normalized_shape = torch.tensor(normalized_shape)
        
        # Learnable parameters gamma and beta
        # Using the same initialization as nn.LayerNorm
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied.
        """
        # We assume normalization is over the last dimension as per standard LayerNorm use case
        # and based on the provided input shapes.
        return triton_layer_norm(x, self.weight, self.bias, self.eps)