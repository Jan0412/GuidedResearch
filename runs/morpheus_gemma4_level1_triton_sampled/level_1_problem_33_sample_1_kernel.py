import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bn_fused_kernel(
    x_ptr, scale_ptr, shift_ptr, out_ptr,
    n_elements,
    C,
    hw,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for fused Batch Normalization.
    Performs the operation: out = x * scale + shift
    where scale and shift are precomputed per channel.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate the channel index for each element in the block.
    # Given input shape (N, C, H, W), the index in the flattened tensor is:
    # idx = n * (C * H * W) + c * (H * W) + h * W + w
    # Therefore, c = (idx // (H * W)) % C
    c_idx = (offsets // hw) % C
    
    # Load the input value and the corresponding channel-wise scale and shift
    x = tl.load(x_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr + c_idx, mask=mask)
    shift = tl.load(shift_ptr + c_idx, mask=mask)
    
    # Perform the fused linear transformation
    out = x * scale + shift
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_batch_norm(x, weight, bias, mean, var, eps):
    """
    Wrapper for the Triton BatchNorm kernel.
    Precomputes the scale and shift parameters to reduce operations inside the kernel.
    """
    # Ensure all tensors are contiguous on GPU
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    
    N, C, H, W = x.shape
    
    # Precompute the fused scale and shift for the operation: y = (x - mean) / sqrt(var + eps) * weight + bias
    # y = x * [weight / sqrt(var + eps)] + [bias - (mean * weight / sqrt(var + eps))]
    # Let scale = weight / sqrt(var + eps)
    # Let shift = bias - mean * scale
    scale = weight / torch.sqrt(var + eps)
    shift = bias - mean * scale
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    hw = H * W
    
    # Tuning parameter for block size
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch the kernel
    bn_fused_kernel[grid](
        x, scale, shift, out,
        n_elements,
        C,
        hw,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that replaces the standard nn.BatchNorm2d with a custom Triton kernel.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.
        """
        super(ModelNew, self).__init__()
        # We keep the BatchNorm2d layer to manage parameters (weight, bias) and buffers (running_mean, running_var)
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor using the optimized Triton kernel.
        """
        # Use the pre-calculated running statistics and parameters from the nn.BatchNorm2d module
        return triton_batch_norm(
            x, 
            self.bn.weight, 
            self.bn.bias, 
            self.bn.running_mean, 
            self.bn.running_var, 
            self.bn.eps
        )