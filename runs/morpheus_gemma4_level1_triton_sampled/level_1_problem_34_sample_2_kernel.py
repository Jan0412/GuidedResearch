import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def instancenorm_kernel(
    x_ptr, 
    y_ptr, 
    w_ptr, 
    b_ptr, 
    N, C, H, W, 
    stride_xn, stride_xc, stride_xw, 
    eps, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one instance (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    # Pointers for this instance
    x_offset = n * stride_xn + c * stride_xc
    ptr = x_ptr + x_offset
    out_ptr = y_ptr + x_offset
    
    # Load weight and bias for this channel
    weight = tl.load(w_ptr + c)
    bias = tl.load(b_ptr + c)

    HW = H * W
    
    # Pass 1: Compute sum and sum of squares for mean and variance
    sum_x = 0.0
    sum_x2 = 0.0
    i = 0
    while i < HW:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        vals = tl.load(ptr + offsets, mask=mask, other=0.0)
        sum_x += tl.sum(vals, axis=0)
        sum_x2 += tl.sum(vals * vals, axis=0)
        i += BLOCK_SIZE

    mean = sum_x / HW
    var = (sum_x2 / HW) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: Normalize and store
    i = 0
    while i < HW:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        vals = tl.load(ptr + offsets, mask=mask, other=0.0)
        # Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
        out = (vals - mean) * inv_std * weight + bias
        tl.store(out_ptr + offsets, out, mask=mask)
        i += BLOCK_SIZE

def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    N, C, H, W = x.shape
    
    out = torch.empty_like(x)
    
    stride_xn = C * H * W
    stride_xc = H * W
    stride_xw = 1
    
    # Grid is one program per instance (N * C)
    grid = (N * C,)
    BLOCK_SIZE = 1024
    
    instancenorm_kernel[grid](
        x, out, weight, bias, 
        N, C, H, W, 
        stride_xn, stride_xc, stride_xw, 
        eps, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using Triton kernels.
    """
    def __init__(self, num_features: int):
        """
        Initializes the Optimized InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        # nn.InstanceNorm2d uses affine=True by default, providing learnable weight and bias
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return triton_instance_norm(x, self.weight, self.bias, self.eps)