import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_stats_kernel(
    x_ptr,
    sum_ptr,
    sum_sq_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute local sum and sum of squares
    sum_val = tl.sum(x)
    sum_sq_val = tl.sum(x * x)
    
    # Atomic add to global sum and sum of squares
    tl.atomic_add(sum_ptr, sum_val)
    tl.atomic_add(sum_sq_ptr, sum_sq_val)


@triton.jit
def layer_norm_apply_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load mean and variance (scalar per batch, broadcasted)
    mean = tl.load(mean_ptr)
    var = tl.load(var_ptr)
    
    # Load weight and bias for the current block
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Compute normalized values
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Applies Layer Normalization using custom Triton kernels.
    
    Args:
        x: Input tensor of shape (*, normalized_shape)
        weight: Weight tensor of shape (normalized_shape,)
        bias: Bias tensor of shape (normalized_shape,)
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized tensor
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get total number of elements in the normalized dimensions
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # Allocate memory for sum and sum of squares
    sum_tensor = torch.zeros((1,), dtype=torch.float32, device=x.device)
    sum_sq_tensor = torch.zeros((1,), dtype=torch.float32, device=x.device)
    
    # Launch stats kernel
    grid_stats = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    layer_norm_stats_kernel[grid_stats](
        x_ptr=x,
        sum_ptr=sum_tensor,
        sum_sq_ptr=sum_sq_tensor,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute mean and variance
    mean = sum_tensor / n_elements
    var = (sum_sq_tensor / n_elements) - mean * mean
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Launch apply kernel
    grid_apply = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    layer_norm_apply_kernel[grid_apply](
        x_ptr=x,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom Triton kernels.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with custom Triton implementation.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-5
        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return triton_layer_norm(x, self.weight, self.bias, self.eps)