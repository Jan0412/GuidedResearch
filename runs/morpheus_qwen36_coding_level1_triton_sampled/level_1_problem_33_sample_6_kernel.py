import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bn_kernel(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    B, C, H, W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    c = tl.program_id(0)
    num_elements = B * H * W
    base_offset = c * H * W
    
    # First pass: compute mean and variance for the channel
    sum_val = 0.0
    sum_sq_val = 0.0
    
    for start in range(0, num_elements, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < num_elements
        
        # Map 1D index to 3D (b, h, w)
        b = idx // (H * W)
        hw = idx % (H * W)
        h = hw // W
        w = hw % W
        
        # Compute pointer offset for NCHW layout
        ptr = x_ptr + b * (C * H * W) + base_offset + h * W + w
        x = tl.load(ptr, mask=mask, other=0.0)
        
        sum_val += tl.sum(x)
        sum_sq_val += tl.sum(x * x)
        
    mean = sum_val / num_elements
    var = sum_sq_val / num_elements - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Load scale and shift parameters for this channel
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    
    # Second pass: apply normalization and affine transformation
    for start in range(0, num_elements, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < num_elements
        
        b = idx // (H * W)
        hw = idx % (H * W)
        h = hw // W
        w = hw % W
        
        ptr_in = x_ptr + b * (C * H * W) + base_offset + h * W + w
        ptr_out = y_ptr + b * (C * H * W) + base_offset + h * W + w
        
        x = tl.load(ptr_in, mask=mask, other=0.0)
        y = (x - mean) * inv_std * gamma + beta
        tl.store(ptr_out, y, mask=mask)


def triton_bn(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Wrapper function to launch the custom BatchNorm2d Triton kernel.
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    
    B, C, H, W = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024  # Tunable block size
    
    # Grid: one block per channel
    grid = (C, 1, 1)
    bn_kernel[grid](x, out, gamma, beta, B, C, H, W, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for Batch Normalization.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replace standard BatchNorm2d forward with custom Triton implementation
        return triton_bn(x, self.bn.weight, self.bn.bias, 1e-5)