import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr
):
    # Compute row index
    row_idx = tl.program_id(0)
    
    # Load input data for this row
    x_row = tl.load(x_ptr + row_idx * N + tl.arange(0, BLOCK_SIZE), mask=row_idx * N + tl.arange(0, BLOCK_SIZE) < M * N, other=0.0)
    
    # Compute mean
    mean = tl.sum(x_row) / N
    
    # Store mean
    tl.store(mean_ptr + row_idx, mean)
    
    # Compute variance
    x_centered = x_row - mean
    var = tl.sum(x_centered * x_centered) / N
    
    # Compute reciprocal standard deviation
    rstd = 1.0 / tl.sqrt(var + EPSILON)
    
    # Store rstd
    tl.store(rstd_ptr + row_idx, rstd)
    
    # Normalize and apply scale and bias
    x_norm = x_centered * rstd
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0)
    
    output = x_norm * weight + bias
    
    # Store output
    tl.store(output_ptr + row_idx * N + tl.arange(0, BLOCK_SIZE), output, mask=row_idx * N + tl.arange(0, BLOCK_SIZE) < M * N)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        # Precompute constants
        self.eps = 1e-5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get dimensions
        batch_size = x.shape[0]
        num_features = x.shape[1] if len(x.shape) > 1 else 1
        total_elements = x.numel()
        
        # Flatten to 2D for processing
        original_shape = x.shape
        x_flat = x.view(-1, num_features)
        
        # Prepare output
        output = torch.empty_like(x_flat)
        
        # Allocate memory for intermediate results
        means = torch.empty(batch_size, dtype=torch.float32, device=x.device)
        rstds = torch.empty(batch_size, dtype=torch.float32, device=x.device)
        
        # Set up kernel launch parameters
        N = num_features
        M = batch_size
        BLOCK_SIZE = 1024  # Tunable parameter
        
        # Determine grid size
        grid = lambda meta: (M,)
        
        # Launch kernel
        layer_norm_kernel[grid](
            x_flat,
            self.weight,
            self.bias,
            output,
            means,
            rstds,
            N,
            M,
            BLOCK_SIZE,
            self.eps
        )
        
        # Reshape output back to original shape
        return output.view(original_shape)