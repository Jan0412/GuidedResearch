import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,          # Pointer to input tensor
    weight_ptr,     # Pointer to gamma
    bias_ptr,       # Pointer to beta
    out_ptr,        # Pointer to output tensor
    n_elements,     # Number of elements in the normalized shape
    eps,            # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # The program ID corresponds to the batch index (row)
    row_idx = tl.program_id(0)
    
    # Pointers for the current row
    x_row_ptr = x_ptr + row_idx * n_elements
    out_row_ptr = out_ptr + row_idx * n_elements

    # --- First Pass: Compute Mean and Variance ---
    sum_val = 0.0
    sq_sum_val = 0.0
    
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_block = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        
        sum_val += tl.sum(x_block * mask)
        sq_sum_val += tl.sum((x_block * x_block) * mask)

    mean = sum_val / n_elements
    # Variance = E[X^2] - (E[X])^2
    var = (sq_sum_val / n_elements) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # --- Second Pass: Normalize, Scale, and Shift ---
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x_block = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        w_block = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        b_block = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        # LayerNorm formula: y = (x - mean) / std * weight + bias
        out_block = (x_block - mean) * inv_std * w_block + b_block
        
        tl.store(out_row_ptr + offsets, out_block, mask=mask)

def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Wrapper for the Triton LayerNorm kernel.
    """
    # Ensure inputs are contiguous and on CUDA
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Input shape: (batch, *normalized_shape)
    # Flatten the normalized dimensions for the kernel
    original_shape = x.shape
    batch_size = original_shape[0]
    n_elements = x.numel() // batch_size
    
    # Reshape to (batch, n_elements) for easier indexing in kernel
    x_flat = x.view(batch_size, n_elements)
    out = torch.empty_like(x_flat)
    
    BLOCK_SIZE = 1024
    # Grid is one program per batch element
    grid = (batch_size,)
    
    layer_norm_kernel[grid](
        x_flat, 
        weight, 
        bias, 
        out, 
        n_elements, 
        eps, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out.view(*original_shape)

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
        # LayerNorm learnable parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied.
        """
        return triton_layer_norm(x, self.weight, self.bias, self.eps)