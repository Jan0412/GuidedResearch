import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def group_norm_kernel(
    x_ptr, out_ptr, gamma_ptr, beta_ptr,
    B, C, H, W, G,
    S, C_per_G, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one group for one batch element
    pid = tl.program_id(0)
    b = pid // G
    g = pid % G
    
    # Calculate the start pointer for the group in the input and output tensors
    # Input x shape: (B, C, H, W). We assume x is contiguous.
    # Stride for B is C * H * W, stride for C is H * W.
    stride_b = C * H * W
    stride_c = H * W
    group_offset = b * stride_b + g * C_per_G * stride_c
    x_group_ptr = x_ptr + group_offset
    out_group_ptr = out_ptr + group_offset
    
    # First pass: Compute mean and variance across the group
    sum_val = 0.0
    sum_sq_val = 0.0
    
    k = 0
    while k < S:
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < S
        vals = tl.load(x_group_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
        sum_sq_val += tl.sum(vals * vals)
        k += BLOCK_SIZE
        
    mean = sum_val / S
    var = (sum_sq_val / S) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Second pass: Normalize and apply scale (gamma) and shift (beta)
    k = 0
    while k < S:
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < S
        vals = tl.load(x_group_ptr + offsets, mask=mask, other=0.0)
        
        # Calculate the global channel index to fetch gamma and beta
        # Local channel index within the group is offset // (H * W)
        c_idx = offsets // HW
        c_global = g * C_per_G + c_idx
        
        gamma = tl.load(gamma_ptr + c_global, mask=mask, other=0.0)
        beta = tl.load(beta_ptr + c_global, mask=mask, other=0.0)
        
        out = (vals - mean) * inv_std * gamma + beta
        tl.store(out_group_ptr + offsets, out, mask=mask)
        k += BLOCK_SIZE

def triton_group_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, eps: float = 1e-5):
    # Ensure inputs are contiguous on GPU
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    
    B, C, H, W = x.shape
    G = num_groups
    C_per_G = C // G
    S = C_per_G * H * W
    HW = H * W
    
    out = torch.empty_like(x)
    
    # Grid: one program per group per batch item
    grid = (B * G,)
    BLOCK_SIZE = 1024
    
    group_norm_kernel[grid](
        x, out, gamma, beta,
        B, C, H, W, G,
        S, C_per_G, HW,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using a custom Triton kernel.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        
        # GroupNorm parameters: weight (gamma) and bias (beta)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied.
        """
        return triton_group_norm(
            x, 
            self.weight, 
            self.bias, 
            self.num_groups, 
            self.eps
        )