import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_forward_kernel(
    X,  # pointer to input tensor
    Y,  # pointer to output tensor
    gamma,  # pointer to scale parameter (optional, can be None)
    beta,  # pointer to bias parameter (optional, can be None)
    mean,  # pointer to precomputed mean
    var,  # pointer to precomputed variance
    C,  # number of channels
    H,  # height
    W,  # width
    eps,  # epsilon for numerical stability
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Each program handles one channel
    c = tl.program_id(0)
    
    # Compute pointer offsets for the current channel
    x_ptr = X + c * H * W
    y_ptr = Y + c * H * W
    
    # Load mean and variance for this channel
    channel_mean = tl.load(mean + c)
    channel_var = tl.load(var + c)
    
    # Compute inverse standard deviation
    inv_std = 1.0 / tl.sqrt(channel_var + eps)
    
    # Load gamma and beta if they exist
    g = tl.load(gamma + c) if gamma is not None else 1.0
    b = tl.load(beta + c) if beta is not None else 0.0
    
    # Iterate over the spatial dimensions in blocks
    for bh in range(0, H, BLOCK_SIZE_H):
        for bw in range(0, W, BLOCK_SIZE_W):
            # Compute offsets for this block
            h_start = bh
            w_start = bw
            h_end = min(bh + BLOCK_SIZE_H, H)
            w_end = min(bw + BLOCK_SIZE_W, W)
            
            # Create meshgrid of offsets
            offsets_h = tl.arange(0, BLOCK_SIZE_H)
            offsets_w = tl.arange(0, BLOCK_SIZE_W)
            mask_h = offsets_h < (h_end - h_start)
            mask_w = offsets_w < (w_end - w_start)
            mask = mask_h[:, None] & mask_w[None, :]
            
            # Compute linear offsets for this block
            offsets = (offsets_h[:, None] * W + offsets_w[None, :])
            offsets = offsets + h_start * W + w_start
            
            # Load input values
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            
            # Apply instance normalization
            x_norm = (x - channel_mean) * inv_std
            out = x_norm * g + b
            
            # Store output
            tl.store(y_ptr + offsets, out, mask=mask)


@triton.jit
def instance_norm_backward_kernel(
    dY,  # pointer to output gradient tensor
    X,  # pointer to input tensor
    gamma,  # pointer to scale parameter (optional, can be None)
    mean,  # pointer to precomputed mean
    var,  # pointer to precomputed variance
    dX,  # pointer to input gradient tensor
    dgamma,  # pointer to gamma gradient (optional, can be None)
    dbeta,  # pointer to beta gradient (optional, can be None)
    C,  # number of channels
    H,  # height
    W,  # width
    eps,  # epsilon for numerical stability
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Each program handles one channel
    c = tl.program_id(0)
    
    # Compute pointer offsets for the current channel
    dy_ptr = dY + c * H * W
    x_ptr = X + c * H * W
    dx_ptr = dX + c * H * W
    
    # Load mean and variance for this channel
    channel_mean = tl.load(mean + c)
    channel_var = tl.load(var + c)
    
    # Compute inverse standard deviation
    inv_std = 1.0 / tl.sqrt(channel_var + eps)
    
    # Load gamma if it exists
    g = tl.load(gamma + c) if gamma is not None else 1.0
    
    # Accumulators for gradients of gamma and beta
    dgamma_sum = 0.0
    dbeta_sum = 0.0
    
    # Iterate over the spatial dimensions in blocks
    for bh in range(0, H, BLOCK_SIZE_H):
        for bw in range(0, W, BLOCK_SIZE_H):
            # Compute offsets for this block
            h_start = bh
            w_start = bw
            h_end = min(bh + BLOCK_SIZE_H, H)
            w_end = min(bw + BLOCK_SIZE_W, W)
            
            # Create meshgrid of offsets
            offsets_h = tl.arange(0, BLOCK_SIZE_H)
            offsets_w = tl.arange(0, BLOCK_SIZE_W)
            mask_h = offsets_h < (h_end - h_start)
            mask_w = offsets_w < (w_end - w_start)
            mask = mask_h[:, None] & mask_w[None, :]
            
            # Compute linear offsets for this block
            offsets = (offsets_h[:, None] * W + offsets_w[None, :])
            offsets = offsets + h_start * W + w_start
            
            # Load input and gradient values
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
            
            # Compute normalized value
            x_norm = (x - channel_mean) * inv_std
            
            # Accumulate gradients for gamma and beta
            dgamma_sum += tl.sum(dy * x_norm * mask)
            dbeta_sum += tl.sum(dy * mask)
            
            # Compute gradient for input
            dx_norm = dy * g
            dvar = tl.sum(dx_norm * (x - channel_mean) * -0.5 * inv_std**3 * mask)
            dmean = tl.sum(dx_norm * -inv_std * mask) + dvar * tl.sum(-2.0 * (x - channel_mean) * mask) / (H * W)
            dx = dx_norm * inv_std + dvar * 2.0 * (x - channel_mean) / (H * W) + dmean / (H * W)
            
            # Store gradient for input
            tl.store(dx_ptr + offsets, dx, mask=mask)
    
    # Store gradients for gamma and beta if they exist
    if dgamma is not None:
        tl.store(dgamma + c, dgamma_sum)
    if dbeta is not None:
        tl.store(dbeta + c, dbeta_sum)


class TritonInstanceNorm2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        # x shape: (batch_size, num_features, height, width)
        B, C, H, W = x.shape
        
        # Compute mean and variance for each channel
        # Reshape to (B*C, H*W) to compute statistics per channel
        x_flat = x.view(B * C, H * W)
        
        # Compute mean
        mean = x_flat.mean(dim=1, keepdim=False)
        
        # Compute variance (unbiased or biased - use biased to match PyTorch)
        var = x_flat.var(dim=1, keepdim=False, correction=0)
        
        # Initialize output tensor
        y = torch.empty_like(x)
        
        # Configure kernel launch parameters
        grid = (C,)
        BLOCK_SIZE_H = 32
        BLOCK_SIZE_W = 32
        
        # Launch kernel
        instance_norm_forward_kernel[grid](
            x, y, weight, bias, mean, var,
            C, H, W, eps,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
        )
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, mean, var)
        ctx.eps = eps
        ctx.C = C
        ctx.H = H
        ctx.W = W
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mean, var = ctx.saved_tensors
        eps = ctx.eps
        C, H, W = ctx.C, ctx.H, ctx.W
        
        # Initialize gradient tensors
        grad_input = torch.empty_like(x)
        grad_weight = torch.zeros_like(weight) if weight is not None else None
        grad_bias = torch.zeros_like(weight) if weight is not None else None
        
        # Configure kernel launch parameters
        grid = (C,)
        BLOCK_SIZE_H = 32
        BLOCK_SIZE_W = 32
        
        # Launch kernel
        instance_norm_backward_kernel[grid](
            grad_output, x, weight, mean, var,
            grad_input, grad_weight, grad_bias,
            C, H, W, eps,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
        )
        
        return grad_input, grad_weight, grad_bias, None


class TritonInstanceNorm2dLayer(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(TritonInstanceNorm2dLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        return TritonInstanceNorm2d.apply(x, self.weight, self.bias, self.eps)


class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using custom Triton kernels.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer with Triton optimization.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.inorm = TritonInstanceNorm2dLayer(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using optimized Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.inorm(x)