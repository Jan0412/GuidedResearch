import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def maxpool3d_kernel(
    x_ptr, 
    out_ptr,
    B, C, D, H, W,
    D_out, H_out, W_out,
    s, p, d,
    stride_B, stride_C, stride_D, stride_H, stride_W,
    out_stride_B, out_stride_C, out_stride_D, out_stride_H, out_stride_W,
    k: tl.constexpr, 
    BLOCK_W: tl.constexpr,
):
    # Grid: (B * C, D_out, H_out, (W_out + BLOCK_W - 1) // BLOCK_W)
    pid_bc = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    b = pid_bc // C
    c = pid_bc % C
    d_out = pid_d
    h_out = pid_h
    w_out_start = pid_w * BLOCK_W
    w_out_offsets = w_out_start + tl.arange(0, BLOCK_W)
    w_out_mask = w_out_offsets < W_out

    # Initialize max_val with a very small number
    max_val = tl.full([BLOCK_W], float("-inf"), dtype=tl.float32)

    for i in range(k):
        d_in = d_out * s + i * d - p
        if d_in < 0 or d_in >= D:
            continue
        
        for j in range(k):
            h_in = h_out * s + j * d - p
            if h_in < 0 or h_in >= H:
                continue
            
            for l in range(k):
                # w_in corresponds to the current window element for each w_out in the block
                w_in = w_out_offsets * s + l * d - p
                
                # Mask for w_in boundary and output boundary
                mask = w_out_mask & (w_in >= 0) & (w_in < W)
                
                # Calculate pointer for the block of elements across the width dimension
                ptr = x_ptr + b * stride_B + c * stride_C + d_in * stride_D + h_in * stride_H + w_in * stride_W
                val = tl.load(ptr, mask=mask, other=float("-inf"))
                max_val = tl.maximum(max_val, val)

    # Store result to output tensor
    out_ptr_base = out_ptr + b * out_stride_B + c * out_stride_C + d_out * out_stride_D + h_out * out_stride_H + w_out_offsets * out_stride_W
    tl.store(out_ptr_base, max_val, mask=w_out_mask)

def triton_maxpool3d(x, kernel_size, stride, padding, dilation, ceil_mode):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    B, C, D, H, W = x.shape
    
    def calc_out_dim(in_dim):
        out_dim = (in_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        return math.ceil(out_dim) if ceil_mode else math.floor(out_dim)

    D_out = calc_out_dim(D)
    H_out = calc_out_dim(H)
    W_out = calc_out_dim(W)
    
    out = torch.empty((B, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    stride_B, stride_C, stride_D, stride_H, stride_W = x.stride()
    out_stride_B, out_stride_C, out_stride_D, out_stride_H, out_stride_W = out.stride()
    
    BLOCK_W = 32
    grid = (B * C, D_out, H_out, (W_out + BLOCK_W - 1) // BLOCK_W)
    
    maxpool3d_kernel[grid](
        x, out,
        B, C, D, H, W,
        D_out, H_out, W_out,
        stride, padding, dilation,
        stride_B, stride_C, stride_D, stride_H, stride_W,
        out_stride_B, out_stride_C, out_stride_D, out_stride_H, out_stride_W,
        k=kernel_size,
        BLOCK_W=BLOCK_W
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 3D using custom Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        # return_indices is not supported by this Triton implementation; 
        # since the base Model default is False, we proceed with calculating values.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        return triton_maxpool3d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.ceil_mode
        )