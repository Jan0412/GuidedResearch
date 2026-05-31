import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    X,  # pointer to input tensor
    Y,  # pointer to output tensor
    Weight,  # pointer to gamma (scale) parameter
    Bias,  # pointer to beta (shift) parameter
    Mean,  # pointer to mean tensor (for inference or backward if needed)
    Var,  # pointer to variance tensor (for inference or backward if needed)
    batch_size,
    num_features,
    num_groups,
    C_per_group,
    HxW,  # product of spatial dimensions (H*W for 2D, etc.)
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one group in one batch element
    # Total number of programs = batch_size * num_groups
    program_id = tl.program_id(0)
    b = program_id // num_groups
    g = program_id % num_groups

    # Compute start and end channels for this group
    start_c = g * C_per_group
    end_c = start_c + C_per_group

    # Compute number of elements in this group: C_per_group * HxW
    elements_per_group = C_per_group * HxW

    # Pointer to start of this batch's data
    X_ptr = X + b * num_features * HxW
    Y_ptr = Y + b * num_features * HxW

    # Accumulators for mean and variance
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over the elements of this group in chunks
    for start in range(0, elements_per_group, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_group
        # Compute global channel offset
        # For each element index within the group, compute which channel it belongs to
        c_idx = offsets // HxW
        spatial_idx = offsets % HxW
        global_c = start_c + c_idx
        # Compute the index in the tensor
        idx = global_c * HxW + spatial_idx
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_val += tl.where(mask, x_f32, 0.0)
        sum_sq_val += tl.where(mask, x_f32 * x_f32, 0.0)

    # Reduce over the BLOCK_SIZE dimension
    sum_total = tl.sum(sum_val, axis=0)
    sum_sq_total = tl.sum(sum_sq_val, axis=0)

    # Compute mean and variance
    mean = sum_total / elements_per_group
    var = sum_sq_total / elements_per_group - mean * mean

    # Compute standard deviation
    std = tl.sqrt(var + eps)

    # Now compute normalized output: (x - mean) / std * weight + bias
    # We need to recompute the indices and do the normalization
    for start in range(0, elements_per_group, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_group
        c_idx = offsets // HxW
        spatial_idx = offsets % HxW
        global_c = start_c + c_idx
        idx = global_c * HxW + spatial_idx

        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        # Normalize
        x_norm = (x_f32 - mean) / std
        # Apply weight and bias (if provided)
        if Weight is not None:
            w = tl.load(Weight + global_c, mask=mask, other=0.0)
            x_norm = x_norm * w
        if Bias is not None:
            b_val = tl.load(Bias + global_c, mask=mask, other=0.0)
            x_norm = x_norm + b_val

        # Store result
        tl.store(Y_ptr + idx, x_norm.to(X.dtype.element_ty), mask=mask)


def triton_group_norm(x, weight, bias, num_groups, eps=1e-5):
    """
    Triton implementation of Group Normalization.
    Args:
        x: Input tensor of shape (batch_size, num_features, *)
        weight: Scale parameter of shape (num_features,)
        bias: Shift parameter of shape (num_features,)
        num_groups: Number of groups
        eps: Small constant for numerical stability
    Returns:
        Normalized tensor with same shape as x
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, num_features = x.shape[0], x.shape[1]
    spatial_shape = x.shape[2:]
    HxW = 1
    for d in spatial_shape:
        HxW *= d
    
    # Check that num_features is divisible by num_groups
    assert num_features % num_groups == 0, f"num_features ({num_features}) must be divisible by num_groups ({num_groups})"
    
    C_per_group = num_features // num_groups
    elements_per_group = C_per_group * HxW
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Grid: one program per (batch, group) pair
    grid = (batch_size * num_groups,)
    
    # Launch kernel
    BLOCK_SIZE = 256
    group_norm_kernel[grid](
        x, out, weight, bias, None, None,
        batch_size, num_features, num_groups,
        C_per_group, HxW, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using Triton kernels.
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
        self.eps = 1e-5
        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return triton_group_norm(x, self.weight, self.bias, self.num_groups, self.eps)


# Configuration for the given example
batch_size = 112  # scaled up
features = 64
num_groups = 8
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features, num_groups] # num_features