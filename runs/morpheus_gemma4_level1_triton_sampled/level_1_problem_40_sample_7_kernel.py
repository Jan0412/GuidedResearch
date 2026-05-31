import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,        # Pointer to input tensor
    weight_ptr,   # Pointer to gamma
    bias_ptr,     # Pointer to beta
    out_ptr,      # Pointer to output tensor
    stride_x_row, # Stride between rows of x
    n_cols,       # Number of elements to normalize over
    eps,          # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one normalization group)
    row_idx = tl.program_id(0)
    
    # Pointers for the current row
    x_row_ptr = x_ptr + row_idx * stride_x_row
    out_row_ptr = out_ptr + row_idx * stride_x_row

    # 1. Compute Mean
    mean = 0.0
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        mean += tl.sum(x)
    mean = mean / n_cols

    # 2. Compute Variance
    var = 0.0
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        var += tl.sum((x - mean) * (x - mean))
    var = var / n_cols

    # 3. Normalize, Scale and Shift
    inv_std = 1.0 / tl.sqrt(var + eps)
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        out = (x - mean) * inv_std * w + b
        tl.store(out_row_ptr + offsets, out, mask=mask)

def triton_layer_norm(x, weight, bias, eps=1e-5):
    # x: (batch, *normalized_shape)
    # weight, bias: normalized_shape
    
    orig_shape = x.shape
    normalized_shape = orig_shape[1:]
    n_cols = 1
    for dim in normalized_shape:
        n_cols *= dim
    
    # Flatten the input to (batch, n_cols)
    x_flat = x.view(-1, n_cols).contiguous()
    weight_flat = weight.view(-1).contiguous()
    bias_flat = bias.view(-1).contiguous()
    
    out_flat = torch.empty_like(x_flat)
    
    batch_size = x_flat.shape[0]
    stride_x_row = x_flat.stride(0)
    
    # We use a large BLOCK_SIZE for the reduction loops
    BLOCK_SIZE = 1024
    
    grid = (batch_size,)
    
    layer_norm_kernel[grid](
        x_flat, weight_flat, bias_flat, out_flat,
        stride_x_row, n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_flat.view(*orig_shape)

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
        
        # Initialize weight and bias parameters to match nn.LayerNorm
        # weight is initialized to ones, bias to zeros
        n_elements = 1
        for dim in normalized_shape:
            n_elements *= dim
            
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Ensure tensors are on CUDA and in FP32 as requested
        x = x.cuda().float()
        weight = self.weight.cuda().float()
        bias = self.bias.cuda().float()
        
        return triton_layer_norm(x, weight, bias, eps=self.eps)