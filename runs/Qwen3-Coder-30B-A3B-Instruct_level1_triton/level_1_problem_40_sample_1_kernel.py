import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Each program processes one row (batch dimension)
    row_start = pid * N
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create full offset for this row
    offsets = row_start + col_offsets
    
    # Load input data
    mask = offsets < (pid + 1) * N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / N
    
    # Store mean for this row
    tl.store(mean_ptr + pid, mean)
    
    # Compute variance
    x_centered = x - mean
    x_squared = x_centered * x_centered
    sum_squared = tl.sum(x_squared, axis=0)
    var = sum_squared / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store reciprocal standard deviation
    tl.store(rstd_ptr + pid, rstd)
    
    # Normalize and apply scale and bias
    x_norm = x_centered * rstd
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    out = x_norm * weight + bias
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with optimized Triton kernel.

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
        Applies optimized Layer Normalization using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Ensure inputs are on GPU
        if not x.is_cuda:
            x = x.cuda()
            
        # Flatten batch dimensions for processing
        original_shape = x.shape
        batch_dims = original_shape[:-len(self.normalized_shape)]
        total_batch_size = 1
        for dim in batch_dims:
            total_batch_size *= dim
            
        # Reshape input to (total_batch_size, *normalized_shape)
        x_reshaped = x.view(total_batch_size, -1)
        
        # Prepare output
        out = torch.empty_like(x_reshaped)
        
        # Prepare buffers for mean and rstd
        mean = torch.empty(total_batch_size, dtype=torch.float32, device=x.device)
        rstd = torch.empty(total_batch_size, dtype=torch.float32, device=x.device)
        
        # Calculate block size
        N = x_reshaped.shape[-1]
        BLOCK_SIZE = 128
        
        # Determine grid size
        grid = (total_batch_size,)
        
        # Launch kernel
        layer_norm_kernel[grid](
            x_reshaped,
            self.weight,
            self.bias,
            out,
            mean,
            rstd,
            N,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape back to original shape
        return out.view(original_shape)