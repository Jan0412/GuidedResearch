import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    X, W, B, Y,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element (one normalization group)
    batch_id = tl.program_id(0)
    
    # Pointers for the current batch element
    x_ptr = X + batch_id * n_elements
    y_ptr = Y + batch_id * n_elements
    
    # 1. Compute Mean
    sum_val = 0.0
    i = 0
    while i < n_elements:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        i += BLOCK_SIZE
    
    mean = sum_val / n_elements
    
    # 2. Compute Variance
    sum_sq_diff = 0.0
    i = 0
    while i < n_elements:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        diff = vals - mean
        sum_sq_diff += tl.sum(diff * diff, axis=0)
        i += BLOCK_SIZE
    
    var = sum_sq_diff / n_elements
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # 3. Normalize, Scale, and Shift
    i = 0
    while i < n_elements:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(W + offsets, mask=mask, other=1.0)
        b = tl.load(B + offsets, mask=mask, other=0.0)
        
        out = (vals - mean) * inv_std * w + b
        tl.store(y_ptr + offsets, out, mask=mask)
        i += BLOCK_SIZE

def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton wrapper for Layer Normalization.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # The normalization is performed over the dimensions specified by the weight/bias shape
    # We flatten the normalization dimensions to treat them as a single vector per batch element
    original_shape = x.shape
    n_elements = weight.numel()
    batch_size = x.numel() // n_elements
    
    x_flat = x.view(batch_size, n_elements).contiguous()
    weight_flat = weight.view(n_elements).contiguous()
    bias_flat = bias.view(n_elements).contiguous()
    
    out_flat = torch.empty_like(x_flat)
    
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    layer_norm_kernel[grid](
        x_flat, weight_flat, bias_flat, out_flat,
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
        self.normalized_shape = torch.tensor(normalized_shape)
        # LayerNorm weights and biases
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return triton_layer_norm(x, self.weight, self.bias, self.eps)