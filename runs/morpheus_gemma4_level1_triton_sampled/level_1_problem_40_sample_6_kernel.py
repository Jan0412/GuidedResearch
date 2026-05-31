import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(
    X, Y, W, B,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one normalization group)
    row_idx = tl.program_id(0)
    
    # Offset pointers to the start of the current row
    X += row_idx * n_elements
    Y += row_idx * n_elements

    # --- Pass 1: Compute Mean ---
    sum_val = 0.0
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_block = tl.load(X + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x_block * mask)
    
    mean = sum_val / n_elements

    # --- Pass 2: Compute Variance ---
    var_val = 0.0
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_block = tl.load(X + offsets, mask=mask, other=0.0)
        diff = x_block - mean
        var_val += tl.sum((diff * diff) * mask)
    
    rstd = 1.0 / tl.sqrt(var_val / n_elements + eps)

    # --- Pass 3: Normalize, Scale, and Shift ---
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_block = tl.load(X + offsets, mask=mask, other=0.0)
        w_block = tl.load(W + offsets, mask=mask, other=1.0)
        b_block = tl.load(B + offsets, mask=mask, other=0.0)
        
        out = (x_block - mean) * rstd * w_block + b_block
        tl.store(Y + offsets, out, mask=mask)

def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    # Ensure inputs are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # The normalization is performed over the trailing dimensions
    # defined by the weight/bias shape.
    n_elements = weight.numel()
    # Reshape x to (batch_size, n_elements)
    original_shape = x.shape
    x_flat = x.view(-1, n_elements)
    batch_size = x_flat.shape[0]
    
    out_flat = torch.empty_like(x_flat)
    
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    layernorm_kernel[grid](
        x_flat, out_flat, weight, bias,
        n_elements,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_flat.view(original_shape)

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
        self.normalized_shape = normalized_shape
        # To maintain parity with nn.LayerNorm, we define weight and bias as parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using the Triton implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied.
        """
        return triton_layernorm(x, self.weight, self.bias, self.eps)