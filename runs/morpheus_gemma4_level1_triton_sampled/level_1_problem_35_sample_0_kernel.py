import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def group_norm_kernel(
    x_ptr, 
    gamma_ptr, 
    beta_ptr, 
    out_ptr, 
    n_elements_per_group, 
    stride_n, 
    stride_c, 
    stride_h, 
    stride_w, 
    c_per_g, 
    hw, 
    eps, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one group for one batch instance
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    # Pointer to the start of the current group
    # x shape: (N, C, H, W)
    # group_ptr = x_ptr + n * (C * H * W) + g * (C_per_G * H * W)
    group_ptr = x_ptr + pid_n * stride_n + pid_g * c_per_g * stride_c
    out_group_ptr = out_ptr + pid_n * stride_n + pid_g * c_per_g * stride_c

    # Pass 1: Compute Mean and Variance
    sum_val = 0.0
    sum_sq_val = 0.0
    
    for k in range(0, n_elements_per_group, BLOCK_SIZE):
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_per_group
        vals = tl.load(group_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
        sum_sq_val += tl.sum(vals * vals)

    mean = sum_val / n_elements_per_group
    var = (sum_sq_val / n_elements_per_group) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: Normalize, Scale, and Shift
    for k in range(0, n_elements_per_group, BLOCK_SIZE):
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_per_group
        vals = tl.load(group_ptr + offsets, mask=mask, other=0.0)

        # Calculate absolute channel index for each element in the block to load gamma and beta
        # c_rel is the channel index relative to the start of the group
        c_rel = offsets // hw
        c_abs = pid_g * c_per_g + c_rel
        
        gamma = tl.load(gamma_ptr + c_abs, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + c_abs, mask=mask, other=0.0)

        # GroupNorm formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
        res = (vals - mean) * inv_std * gamma + beta
        tl.store(out_group_ptr + offsets, res, mask=mask)


def triton_group_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    n, c, h, w = x.shape
    c_per_g = c // num_groups
    hw = h * w
    n_elements_per_group = c_per_g * hw
    
    out = torch.empty_like(x)

    # Strides for the input tensor
    stride_n = c * h * w
    stride_c = h * w
    stride_h = w
    stride_w = 1
    
    eps = 1e-5
    BLOCK_SIZE = 1024

    # Grid: (batch_size, num_groups)
    grid = (n, num_groups)

    group_norm_kernel[grid](
        x, gamma, beta, out,
        n_elements_per_group,
        stride_n, stride_c, stride_h, stride_w,
        c_per_g, hw, eps,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return triton_group_norm(x, self.weight, self.bias, self.num_groups)