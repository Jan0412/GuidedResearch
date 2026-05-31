import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,          # Pointer to input tensor
    gamma_ptr,      # Pointer to gamma (weight)
    beta_ptr,       # Pointer to beta (bias)
    out_ptr,        # Pointer to output tensor
    n_elements,     # Number of elements to normalize per row (product of normalized_shape)
    stride_row,     # Stride between rows (batch dimension)
    eps,            # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one sample in the batch)
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * stride_row
    
    # Pass 1: Compute mean and variance
    sum_x = 0.0
    sum_x2 = 0.0
    
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        # Load input values
        x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
    
    mean = sum_x / n_elements
    # Variance = E[X^2] - (E[X])^2
    var = (sum_x2 / n_elements) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Pass 2: Normalize, scale, and shift
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load values
        x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
        
        # Apply normalization: (x - mean) * inv_std * gamma + beta
        out = (x - mean) * inv_std * gamma + beta
        
        # Store result
        tl.store(out_ptr + row_idx * stride_row + offsets, out, mask=mask)

def triton_layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    """
    Wrapper for the Triton LayerNorm kernel.
    """
    # Ensure inputs are contiguous and on GPU
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    
    # Input shape: (B, ...) where ... is the normalized_shape
    B = x.shape[0]
    n_elements = x[0].numel()
    stride_row = n_elements
    
    out = torch.empty_like(x)
    
    # Block size for the reduction and application loops
    BLOCK_SIZE = 1024
    
    # Grid: one program per batch item
    grid = (B,)
    
    layer_norm_kernel[grid](
        x, gamma, beta, out,
        n_elements, stride_row, eps,
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
        self.normalized_shape = torch.tensor(normalized_shape)
        # Weight and bias parameters, matching nn.LayerNorm initialization
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return triton_layer_norm(x, self.gamma, self.beta, self.eps)