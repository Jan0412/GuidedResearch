import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    X_ptr,  # Pointer to input tensor
    Y_ptr,  # Pointer to output tensor
    W_ptr,  # Pointer to weight tensor (gamma)
    B_ptr,  # Pointer to bias tensor (beta)
    M_ptr,  # Pointer to mean
    V_ptr,  # Pointer to variance
    N,      # Number of elements to normalize in each group (last dimensions)
    C,      # Total number of groups (batch_size * features * other_dims before normalized_shape)
    eps,    # Small epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one group (one layer norm operation)
    group_id = tl.program_id(0)
    
    # Compute the starting index for this group
    x_offset = group_id * N
    
    # Initialize statistics for layer norm
    sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute mean
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(X_ptr + x_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        sum += tl.where(mask, x, 0.0)
        sum_sq += tl.where(mask, x * x, 0.0)
    
    # Reduce across the block dimension
    mean = tl.sum(sum) / N
    var = tl.sum(sum_sq) / N - mean * mean
    
    # Store mean and variance
    tl.store(M_ptr + group_id, mean)
    tl.store(V_ptr + group_id, var)
    
    # Normalize and scale
    rstd = 1.0 / tl.sqrt(var + eps)
    
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        x = tl.load(X_ptr + x_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        x_norm = (x - mean) * rstd
        
        # Load weight and bias
        w = tl.load(W_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Apply scaling and shifting
        out = x_norm * w + b
        
        # Store result
        tl.store(Y_ptr + x_offset + offsets, out.to(X_ptr.dtype.element_ty), mask=mask)


def triton_layer_norm(x: torch.Tensor, normalized_shape: tuple, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of Layer Normalization.
    
    Args:
        x: Input tensor of shape (*, normalized_shape)
        normalized_shape: Shape of the normalization dimensions
        weight: Gamma parameter of shape normalized_shape
        bias: Beta parameter of shape normalized_shape
        eps: Small value for numerical stability
        
    Returns:
        Output tensor with same shape as input
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get input shape
    input_shape = x.shape
    *batch_dims, norm_shape = input_shape[:-len(normalized_shape)], input_shape[-len(normalized_shape):]
    
    # Verify normalized_shape matches
    assert norm_shape == normalized_shape, f"Normalized shape mismatch: {norm_shape} vs {normalized_shape}"
    
    # Compute total number of groups and elements per group
    C = 1
    for d in batch_dims:
        C *= d
    N = 1
    for d in normalized_shape:
        N *= d
    
    # Flatten batch dimensions for easier processing
    x_flat = x.view(C, N)
    
    # Prepare output tensors
    y = torch.empty_like(x)
    mean = torch.empty((C,), device=x.device, dtype=torch.float32)
    var = torch.empty((C,), device=x.device, dtype=torch.float32)
    
    # Define block size (tunable parameter)
    BLOCK_SIZE = min(256, triton.next_power_of_2(N))
    
    # Define grid
    grid = (C,)
    
    # Launch kernel
    layer_norm_kernel[grid](
        x_flat, y.view(C, N), weight, bias, mean, var,
        N, C, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape output to original shape
    y = y.view(*input_shape)
    
    return y, mean, var


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom Triton kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with Triton optimization.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        # Initialize weight and bias parameters
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Call our Triton-based layer norm
        y, _, _ = triton_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return y