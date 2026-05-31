import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    X,  # Pointer to input tensor
    Y,  # Pointer to output tensor
    W,  # Pointer to weight tensor (gamma)
    B,  # Pointer to bias tensor (beta)
    Mean,  # Pointer to mean values
    Rstd,  # Pointer to reciprocal standard deviation values
    N,  # Number of elements to normalize per sample
    C,  # Total number of samples in batch
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Get the sample index this program instance will process
    sample_id = tl.program_id(0)
    
    # Calculate offset to start of this sample
    x_ptr = X + sample_id * N
    y_ptr = Y + sample_id * N
    
    # Initialize accumulators for mean calculation
    sum_x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load and accumulate for mean
    for start_n in range(0, N, BLOCK_SIZE):
        offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_x += x.to(tl.float32)
    
    # Compute mean
    mean = tl.sum(sum_x, axis=0) / N
    tl.store(Mean + sample_id, mean)
    
    # Compute variance and standard deviation
    sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_SIZE):
        offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        x_centered = x.to(tl.float32) - mean
        sum_sq += x_centered * x_centered
    
    var = tl.sum(sum_sq, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + sample_id, rstd)
    
    # Normalize, scale, and shift in a single pass
    for start_n in range(0, N, BLOCK_SIZE):
        offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        # Load input
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize
        x_centered = x.to(tl.float32) - mean
        x_norm = x_centered * rstd
        
        # Scale and shift
        w = tl.load(W + offsets, mask=mask, other=1.0)
        b = tl.load(B + offsets, mask=mask, other=0.0)
        y = x_norm * w.to(tl.float32) + b.to(tl.float32)
        
        # Store output
        tl.store(y_ptr + offsets, y.to(tl.float32), mask=mask)


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Triton implementation of LayerNorm.
    
    Args:
        x: Input tensor of shape (C, N) where C is batch size and N is number of elements to normalize
        weight: Scale parameter of shape (N,)
        bias: Shift parameter of shape (N,)
        eps: Small value for numerical stability
    
    Returns:
        Normalized tensor of same shape as x
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get dimensions
    C = x.shape[0]
    N = x.shape[1]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Allocate temporary storage for mean and rstd
    mean = torch.empty(C, dtype=torch.float32, device=x.device)
    rstd = torch.empty(C, dtype=torch.float32, device=x.device)
    
    # Determine block size (tunable parameter)
    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    
    # Grid: one block per sample
    grid = (C,)
    
    # Launch kernel
    layer_norm_kernel[grid](
        x, out, weight, bias, mean, rstd,
        N, C, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the optimized LayerNorm layer.
        
        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        
        # Store normalized shape
        self.normalized_shape = normalized_shape
        self.eps = 1e-5
        
        # Calculate total number of elements to normalize per sample
        n_elements = 1
        for dim in normalized_shape:
            n_elements *= dim
        
        # Initialize weight and bias parameters (gamma and beta)
        self.weight = nn.Parameter(torch.ones(n_elements))
        self.bias = nn.Parameter(torch.zeros(n_elements))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).
        
        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied.
        """
        # Save original shape
        original_shape = x.shape
        
        # Flatten all dimensions except the last normalized_shape dimensions
        batch_size = 1
        for dim in original_shape[:-len(self.normalized_shape)]:
            batch_size *= dim
            
        # Reshape to (batch_size, normalized_elements)
        x_flat = x.view(batch_size, -1)
        
        # Apply LayerNorm
        out_flat = triton_layer_norm(x_flat, self.weight, self.bias, self.eps)
        
        # Reshape back to original shape
        out = out_flat.view(original_shape)
        
        return out