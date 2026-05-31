import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    X_ptr,  # Pointer to input tensor
    Y_ptr,  # Pointer to output tensor
    W_ptr,  # Pointer to weight tensor (optional, not used here as RMSNorm doesn't have learnable params)
    batch_size,  # Batch size
    num_features,  # Number of features
    dim1,  # First spatial dimension
    dim2,  # Second spatial dimension
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sample in the batch
    batch_idx = tl.program_id(0)
    
    # Calculate offset to the start of this batch's data
    # Each batch has shape (num_features, dim1, dim2)
    batch_offset = batch_idx * num_features * dim1 * dim2
    
    # We'll compute the RMS for each feature in this batch
    for feat_idx in range(num_features):
        # Calculate offset to this feature
        feat_offset = batch_offset + feat_idx * dim1 * dim2
        
        # Compute sum of squares for this feature
        sum_sq = 0.0
        for i in range(dim1):
            for j in range(dim2):
                # Calculate the actual offset to the element
                element_offset = feat_offset + i * dim2 + j
                x_val = tl.load(X_ptr + element_offset)
                sum_sq += x_val * x_val
        
        # Compute mean of squares
        mean_sq = sum_sq / (dim1 * dim2)
        
        # Compute RMS with epsilon
        rms = tl.sqrt(mean_sq + eps)
        
        # Normalize and store the result
        for i in range(dim1):
            for j in range(dim2):
                element_offset = feat_offset + i * dim2 + j
                x_val = tl.load(X_ptr + element_offset)
                y_val = x_val / rms
                tl.store(Y_ptr + element_offset, y_val)


def triton_rms_norm(x: torch.Tensor, eps: float = 1e-5):
    """
    Applies RMS Normalization using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, num_features, dim1, dim2)
        eps: Small value for numerical stability
        
    Returns:
        Normalized tensor of same shape as input
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    batch_size, num_features, dim1, dim2 = x.shape
    
    # Grid: one program per batch element
    grid = (batch_size,)
    
    # Launch the Triton kernel
    rms_norm_kernel[grid](
        x, out, None,  # X_ptr, Y_ptr, W_ptr
        batch_size, num_features, dim1, dim2, 
        eps,
        BLOCK_SIZE=128  # Not used in this implementation but kept for structure
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using Triton kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, dim1, dim2).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_rms_norm(x, self.eps)