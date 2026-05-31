import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batchnorm_forward_kernel(
    x_ptr,  # Input tensor [B, C, H, W]
    weight_ptr,  # Scale parameter [C]
    bias_ptr,  # Shift parameter [C]
    running_mean_ptr,  # Running mean [C]
    running_var_ptr,  # Running variance [C]
    save_mean_ptr,  # Saved mean for backward [C]
    save_inv_std_ptr,  # Saved inverse std for backward [C]
    out_ptr,  # Output tensor [B, C, H, W]
    batch_size,  # B
    num_features,  # C
    spatial_size,  # H * W
    eps,  # Epsilon for numerical stability
    is_training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature channel
    feat_id = tl.program_id(0)
    
    # Compute statistics for this channel
    if is_training:
        # Compute mean
        sum_val = 0.0
        for b in range(batch_size):
            for i in range(spatial_size):
                offset = b * num_features * spatial_size + feat_id * spatial_size + i
                val = tl.load(x_ptr + offset)
                sum_val += val
        
        mean = sum_val / (batch_size * spatial_size)
        
        # Compute variance
        var_sum = 0.0
        for b in range(batch_size):
            for i in range(spatial_size):
                offset = b * num_features * spatial_size + feat_id * spatial_size + i
                val = tl.load(x_ptr + offset)
                var_sum += (val - mean) ** 2
        
        var = var_sum / (batch_size * spatial_size)
        inv_std = 1.0 / tl.sqrt(var + eps)
        
        # Save for backward pass
        tl.store(save_mean_ptr + feat_id, mean)
        tl.store(save_inv_std_ptr + feat_id, inv_std)
    else:
        mean = tl.load(running_mean_ptr + feat_id)
        var = tl.load(running_var_ptr + feat_id)
        inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Apply normalization and affine transformation
    weight = tl.load(weight_ptr + feat_id)
    bias = tl.load(bias_ptr + feat_id)
    
    for b in range(batch_size):
        for i in range(spatial_size):
            offset = b * num_features * spatial_size + feat_id * spatial_size + i
            val = tl.load(x_ptr + offset)
            normalized = (val - mean) * inv_std
            out_val = normalized * weight + bias
            tl.store(out_ptr + offset, out_val)


@triton.jit
def batchnorm_backward_kernel(
    grad_output_ptr,  # Gradient of output [B, C, H, W]
    x_ptr,  # Input tensor [B, C, H, W]
    weight_ptr,  # Scale parameter [C]
    save_mean_ptr,  # Saved mean [C]
    save_inv_std_ptr,  # Saved inverse std [C]
    grad_weight_ptr,  # Gradient of weight [C]
    grad_bias_ptr,  # Gradient of bias [C]
    grad_input_ptr,  # Gradient of input [B, C, H, W]
    batch_size,  # B
    num_features,  # C
    spatial_size,  # H * W
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature channel
    feat_id = tl.program_id(0)
    
    # Compute gradients for weight and bias
    grad_weight_sum = 0.0
    grad_bias_sum = 0.0
    
    mean = tl.load(save_mean_ptr + feat_id)
    inv_std = tl.load(save_inv_std_ptr + feat_id)
    weight = tl.load(weight_ptr + feat_id)
    
    # First pass: compute gradients for weight and bias
    for b in range(batch_size):
        for i in range(spatial_size):
            offset = b * num_features * spatial_size + feat_id * spatial_size + i
            grad_out = tl.load(grad_output_ptr + offset)
            val = tl.load(x_ptr + offset)
            normalized = (val - mean) * inv_std
            
            grad_weight_sum += grad_out * normalized
            grad_bias_sum += grad_out
    
    # Store gradients for weight and bias
    tl.store(grad_weight_ptr + feat_id, grad_weight_sum)
    tl.store(grad_bias_ptr + feat_id, grad_bias_sum)
    
    # Second pass: compute gradient for input
    # Precompute constants for the gradient computation
    grad_weight = tl.load(grad_weight_ptr + feat_id)
    grad_bias = tl.load(grad_bias_ptr + feat_id)
    
    # Compute the gradient contribution
    for b in range(batch_size):
        for i in range(spatial_size):
            offset = b * num_features * spatial_size + feat_id * spatial_size + i
            grad_out = tl.load(grad_output_ptr + offset)
            val = tl.load(x_ptr + offset)
            normalized = (val - mean) * inv_std
            
            # Gradient of batchnorm: 
            # dL/dx = (dL/dy * weight) * inv_std 
            #       - (dL/dy * weight * inv_std * normalized) * (sum(normalized) / N) 
            #       - (dL/dy * weight * inv_std) * (sum(1) / N)
            # This can be simplified to:
            # dL/dx = (dL/dy * weight - mean_grad_weight * normalized - mean_grad_bias) * inv_std
            
            # Compute mean of grad_output * weight * normalized
            # and mean of grad_output * weight
            # But since we're doing it per-thread, we need to use the precomputed sums
            # Actually, let's recompute with the correct formula
            
            # For now, we'll use a simpler approach - compute the full gradient
            # This requires knowing the means, so we'll do it in a separate kernel or pass
            
            # Actually, let's use the correct formula:
            # dL/dx = (1 / (N*H*W)) * inv_std * 
            #         (N*H*W * grad_out * weight 
            #          - sum(grad_out * weight) * normalized 
            #          - sum(grad_out * weight * normalized) * normalized 
            #          - sum(grad_out) * normalized)
            
            # But this is complex to implement in a single kernel pass, so let's 
            # use a more efficient approach by computing the necessary means
            
            # For simplicity, we'll compute the gradient with the precomputed sums
            # But we need to restructure this - let's do it in two passes
            
            pass  # We'll implement this properly below


@triton.jit
def batchnorm_backward_kernel_v2(
    grad_output_ptr,  # Gradient of output [B, C, H, W]
    x_ptr,  # Input tensor [B, C, H, W]
    weight_ptr,  # Scale parameter [C]
    save_mean_ptr,  # Saved mean [C]
    save_inv_std_ptr,  # Saved inverse std [C]
    grad_weight_ptr,  # Gradient of weight [C]
    grad_bias_ptr,  # Gradient of bias [C]
    grad_input_ptr,  # Gradient of input [B, C, H, W]
    batch_size,  # B
    num_features,  # C
    spatial_size,  # H * W
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature channel
    feat_id = tl.program_id(0)
    
    mean = tl.load(save_mean_ptr + feat_id)
    inv_std = tl.load(save_inv_std_ptr + feat_id)
    weight = tl.load(weight_ptr + feat_id)
    
    # Compute sums needed for gradient computation
    sum1 = 0.0  # sum(grad_output * weight * normalized)
    sum2 = 0.0  # sum(grad_output * weight)
    sum3 = 0.0  # sum(grad_output)
    
    for b in range(batch_size):
        for i in range(spatial_size):
            offset = b * num_features * spatial_size + feat_id * spatial_size + i
            grad_out = tl.load(grad_output_ptr + offset)
            val = tl.load(x_ptr + offset)
            normalized = (val - mean) * inv_std
            
            sum1 += grad_out * weight * normalized
            sum2 += grad_out * weight
            sum3 += grad_out
    
    # Compute gradients for weight and bias
    tl.store(grad_weight_ptr + feat_id, sum1)
    tl.store(grad_bias_ptr + feat_id, sum3)
    
    # Compute gradient for input
    N = batch_size * spatial_size
    for b in range(batch_size):
        for i in range(spatial_size):
            offset = b * num_features * spatial_size + feat_id * spatial_size + i
            grad_out = tl.load(grad_output_ptr + offset)
            val = tl.load(x_ptr + offset)
            normalized = (val - mean) * inv_std
            
            # BatchNorm backward formula:
            # dL/dx = (1/N) * inv_std * (N * grad_out * weight - sum2 - sum1 * normalized)
            grad_input = (grad_out * weight - sum2 / N - sum1 * normalized / N) * inv_std
            tl.store(grad_input_ptr + offset, grad_input)


class TritonBatchNorm2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, training, eps):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        running_mean = running_mean.contiguous()
        running_var = running_var.contiguous()
        
        batch_size, num_features, dim1, dim2 = x.shape
        spatial_size = dim1 * dim2
        
        # Allocate output tensor
        out = torch.empty_like(x)
        
        # Allocate save statistics for backward pass
        save_mean = torch.empty(num_features, device=x.device, dtype=x.dtype)
        save_inv_std = torch.empty(num_features, device=x.device, dtype=x.dtype)
        
        # Launch kernel
        grid = (num_features,)
        batchnorm_forward_kernel[grid](
            x, weight, bias, running_mean, running_var,
            save_mean, save_inv_std, out,
            batch_size, num_features, spatial_size, eps,
            is_training=training,
            BLOCK_SIZE=256
        )
        
        # Save for backward pass
        if training:
            ctx.save_for_backward(x, weight, save_mean, save_inv_std)
            ctx.batch_size = batch_size
            ctx.num_features = num_features
            ctx.spatial_size = spatial_size
            ctx.eps = eps
        
        return out, save_mean, save_inv_std
    
    @staticmethod
    def backward(ctx, grad_output, grad_save_mean, grad_save_inv_std):
        x, weight, save_mean, save_inv_std = ctx.saved_tensors
        batch_size = ctx.batch_size
        num_features = ctx.num_features
        spatial_size = ctx.spatial_size
        
        # Ensure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        # Allocate gradient tensors
        grad_input = torch.empty_like(x)
        grad_weight = torch.empty(num_features, device=x.device, dtype=x.dtype)
        grad_bias = torch.empty(num_features, device=x.device, dtype=x.dtype)
        
        # Launch backward kernel
        grid = (num_features,)
        batchnorm_backward_kernel_v2[grid](
            grad_output, x, weight, save_mean, save_inv_std,
            grad_weight, grad_bias, grad_input,
            batch_size, num_features, spatial_size,
            BLOCK_SIZE=256
        )
        
        return grad_input, grad_weight, grad_bias, None, None, None, None


class TritonBatchNorm2d(nn.Module):
    """
    Custom BatchNorm2d implementation using Triton kernels.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(TritonBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
    
    def forward(self, x):
        if self.training and self.track_running_stats:
            # Use the current batch statistics
            num_features = self.num_features
            batch_size, c, dim1, dim2 = x.shape
            assert c == num_features, f"Expected {num_features} channels, got {c}"
            
            # Convert to contiguous
            x = x.contiguous()
            
            # Allocate outputs
            out = torch.empty_like(x)
            save_mean = torch.empty(num_features, device=x.device, dtype=x.dtype)
            save_inv_std = torch.empty(num_features, device=x.device, dtype=x.dtype)
            
            # Run kernel
            grid = (num_features,)
            batchnorm_forward_kernel[grid](
                x, self.weight, self.bias, 
                self.running_mean, self.running_var,
                save_mean, save_inv_std, out,
                batch_size, num_features, dim1 * dim2, self.eps,
                is_training=True,
                BLOCK_SIZE=256
            )
            
            # Update running statistics
            with torch.no_grad():
                # Compute batch statistics for running stats update
                batch_mean = x.mean(dim=[0, 2, 3])
                batch_var = x.var(dim=[0, 2, 3], unbiased=False)
                
                # Update running_mean and running_var
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    momentum = 1.0 / self.num_batches_tracked.item() if self.num_batches_tracked.item() > 0 else 1.0
                else:
                    momentum = self.momentum
                
                # Use the momentum parameter as in PyTorch
                if self.momentum is None:
                    # cumulative moving average
                    self.running_mean.mul_(1 - 1.0 / self.num_batches_tracked).add_(batch_mean / self.num_batches_tracked)
                    self.running_var.mul_(1 - 1.0 / self.num_batches_tracked).add_(batch_var / self.num_batches_tracked)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
            
            return out
        
        else:
            # Use running statistics for inference or evaluation
            num_features = self.num_features
            batch_size, c, dim1, dim2 = x.shape
            assert c == num_features, f"Expected {num_features} channels, got {c}"
            
            # Convert to contiguous
            x = x.contiguous()
            
            # Allocate outputs
            out = torch.empty_like(x)
            
            # Run kernel
            grid = (num_features,)
            batchnorm_forward_kernel[grid](
                x, self.weight, self.bias, 
                self.running_mean, self.running_var,
                None, None, out,
                batch_size, num_features, dim1 * dim2, self.eps,
                is_training=False,
                BLOCK_SIZE=256
            )
            
            return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for BatchNorm2d.
    """
    def __init__(self, num_features: int):
        """
        Initializes the optimized BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.bn = TritonBatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(x)