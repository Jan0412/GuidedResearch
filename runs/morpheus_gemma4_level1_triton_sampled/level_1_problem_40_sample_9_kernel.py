import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    X, Y, W, B, 
    N, 
    stride_x_batch, 
    stride_y_batch,
    epsilon,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one batch item
    batch_id = tl.program_id(0)
    
    # Pointers for the current batch item
    x_ptr = X + batch_id * stride_x_batch
    y_ptr = Y + batch_id * stride_y_batch
    
    # 1. Compute Mean
    sum_val = 0.0
    i = 0
    while i < N:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        i += BLOCK_SIZE
    
    mean = sum_val / N
    
    # 2. Compute Variance
    sq_sum_val = 0.0
    i = 0
    while i < N:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        diff = vals - mean
        sq_sum_val += tl.sum(diff * diff, axis=0)
        i += BLOCK_SIZE
    
    var = sq_sum_val / N
    rstd = 1.0 / tl.sqrt(var + epsilon)
    
    # 3. Normalize, Scale and Shift
    i = 0
    while i < N:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        w = tl.load(W + offsets, mask=mask, other=1.0)
        b = tl.load(B + offsets, mask=mask, other=0.0)
        
        out = (vals - mean) * rstd * w + b
        tl.store(y_ptr + offsets, out, mask=mask)
        i += BLOCK_SIZE

def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, epsilon: float = 1e-5):
    # Ensure inputs are contiguous and FP32
    x = x.contiguous().float()
    weight = weight.contiguous().float()
    bias = bias.contiguous().float()
    
    original_shape = x.shape
    # Flatten the normalization dimensions
    # x shape: (batch_size, *normalized_shape)
    batch_size = original_shape[0]
    normalized_shape = original_shape[1:]
    
    # Total elements in the normalization shape
    N = 1
    for dim in normalized_shape:
        N *= dim
        
    # Flatten x to (batch_size, N)
    x_flat = x.view(batch_size, N)
    out_flat = torch.empty_like(x_flat)
    
    # Weight and bias are normalized_shape, flatten to (N,)
    weight_flat = weight.view(-1)
    bias_flat = bias.view(-1)
    
    # Kernel configuration
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    layer_norm_kernel[grid](
        x_flat, 
        out_flat, 
        weight_flat, 
        bias_flat, 
        N, 
        x_flat.stride(0), 
        out_flat.stride(0), 
        epsilon, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_flat.view(original_shape)

class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom Triton kernels.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = torch.tensor(normalized_shape)
        # Initialize parameters similarly to nn.LayerNorm
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.epsilon = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return triton_layer_norm(x, self.weight, self.bias, self.epsilon)