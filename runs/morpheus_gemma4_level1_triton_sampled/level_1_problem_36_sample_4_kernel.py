import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, 
    out_ptr, 
    stride_b, 
    stride_f, 
    stride_d1, 
    stride_d2, 
    B, 
    F, 
    D1, 
    D2, 
    eps, 
    BLOCK_F: tl.constexpr,
):
    # Each program handles one normalization group (B, D1, D2)
    group_id = tl.program_id(0)
    
    # Decompose group_id into coordinates (b, d1, d2)
    # group_id = b * (D1 * D2) + d1 * D2 + d2
    d2 = group_id % D2
    rem = group_id // D2
    d1 = rem % D1
    b = rem // D1
    
    # Calculate the base offset for the first element of the group (f=0)
    base_offset = b * stride_b + d1 * stride_d1 + d2 * stride_d2
    
    # Create offsets for the feature dimension F
    offsets = tl.arange(0, BLOCK_F)
    mask = offsets < F
    
    # Load the elements of the group: x[b, f, d1, d2]
    # The distance between elements in the feature dimension is stride_f
    ptr = x_ptr + base_offset + offsets * stride_f
    vals = tl.load(ptr, mask=mask, other=0.0)
    
    # Compute the sum of squares: sum(x^2)
    sq_sum = tl.sum(vals * vals, axis=0)
    
    # Compute the Root Mean Square (RMS)
    # RMS = sqrt(mean(x^2) + eps)
    rms = tl.sqrt(sq_sum / F + eps)
    
    # Normalize the values
    out = vals / rms
    
    # Store the result back to the output tensor
    out_ptr_group = out_ptr + base_offset + offsets * stride_f
    tl.store(out_ptr_group, out, mask=mask)

def triton_rms_norm(x: torch.Tensor, eps: float):
    """
    Triton wrapper for RMS Normalization.
    """
    # Ensure input is on CUDA
    assert x.is_cuda, "Tensors must be on CUDA."
    
    B, F, D1, D2 = x.shape
    out = torch.empty_like(x)
    
    # Get strides of the input tensor
    stride_b, stride_f, stride_d1, stride_d2 = x.stride()
    
    # Determine the block size for the feature dimension (must be power of 2)
    BLOCK_F = triton.next_power_of_2(F)
    
    # The grid is defined by the non-reduction dimensions
    grid = (B * D1 * D2,)
    
    rms_norm_kernel[grid](
        x, 
        out, 
        stride_b, 
        stride_f, 
        stride_d1, 
        stride_d2, 
        B, 
        F, 
        D1, 
        D2, 
        eps, 
        BLOCK_F=BLOCK_F
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using a custom Triton kernel.
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
        Applies RMS Normalization to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return triton_rms_norm(x, self.eps)