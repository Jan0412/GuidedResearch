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
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate the starting offset for this row
    row_start = row_idx * N
    
    # Load input data for this row
    x_row = tl.load(x_ptr + row_start + tl.arange(0, BLOCK_SIZE), mask=row_start + tl.arange(0, BLOCK_SIZE) < row_start + N, other=0.0)
    
    # Compute mean
    mean = tl.sum(x_row) / N
    
    # Store mean for this row
    tl.store(mean_ptr + row_idx, mean)
    
    # Compute variance
    x_centered = x_row - mean
    variance = tl.sum(x_centered * x_centered) / N
    
    # Compute reciprocal standard deviation with epsilon
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    # Store reciprocal standard deviation
    tl.store(rstd_ptr + row_idx, rstd)
    
    # Normalize and scale
    normalized = x_centered * rstd
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0)
    
    # Apply scale and bias
    output = normalized * weight + bias
    
    # Store output
    tl.store(output_ptr + row_start + tl.arange(0, BLOCK_SIZE), output, mask=row_start + tl.arange(0, BLOCK_SIZE) < row_start + N)

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
        
        # Flatten input for processing
        original_shape = x.shape
        batch_size = x.shape[0]
        num_elements = x.numel() // batch_size
        
        # Reshape to 2D for processing
        x_flat = x.view(batch_size, -1)
        
        # Allocate output tensor
        output = torch.empty_like(x_flat)
        
        # Allocate mean and rstd tensors
        mean = torch.empty(batch_size, dtype=torch.float32, device=x.device)
        rstd = torch.empty(batch_size, dtype=torch.float32, device=x.device)
        
        # Set up kernel launch parameters
        BLOCK_SIZE = 1024
        grid = (batch_size,)
        
        # Launch kernel
        layer_norm_kernel[grid](
            x_flat,
            self.weight,
            self.bias,
            output,
            mean,
            rstd,
            num_elements,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape back to original shape
        return output.view(original_shape)