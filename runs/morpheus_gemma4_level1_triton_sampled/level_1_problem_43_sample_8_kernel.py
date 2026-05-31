import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def maxpool3d_kernel(
    x_ptr, 
    out_ptr, 
    N, C, D, H, W, 
    OD, OH, OW, 
    K, S, P, Dil, 
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w
):
    # Map program ID to output coordinates
    n = tl.program_id(0)
    c = tl.program_id(1)
    d = tl.program_id(2)
    h = tl.program_id(3)
    w = tl.program_id(4)

    # Calculate the output pointer offset
    out_offset = (n * out_stride_n + 
                  c * out_stride_c + 
                  d * out_stride_d + 
                  h * out_stride_h + 
                  w * out_stride_w)
    
    # Calculate the base input pointer offset for the current batch and channel
    x_base_offset = n * stride_n + c * stride_c

    # Initialize max value to negative infinity
    max_val = -float('inf')

    # Iterate over the 3D pooling window
    for kd in range(0, K):
        id_val = d * S - P + kd * Dil
        if id_val < 0 or id_val >= D:
            continue
        
        d_offset = id_val * stride_d
        for kh in range(0, K):
            ih_val = h * S - P + kh * Dil
            if ih_val < 0 or ih_val >= H:
                continue
            
            h_offset = ih_val * stride_h
            for kw in range(0, K):
                iw_val = w * S - P + kw * Dil
                if iw_val < 0 or iw_val >= W:
                    continue
                
                w_offset = iw_val * stride_w
                
                # Load value from input tensor
                val = tl.load(x_ptr + x_base_offset + d_offset + h_offset + w_offset)
                max_val = tl.maximum(max_val, val)

    # Store the result in the output tensor
    tl.store(out_ptr + out_offset, max_val)


def triton_maxpool3d(x, kernel_size, stride, padding, dilation, ceil_mode=False):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    N, C, D, H, W = x.shape
    S = stride if stride is not None else kernel_size
    
    # Calculate output dimensions
    def calc_out_dim(dim, k, s, p, dil, ceil):
        out = (dim + 2 * p - dil * (k - 1) - 1)
        if ceil:
            return math.ceil(out / s) + 1
        else:
            return (out // s) + 1

    OD = calc_out_dim(D, kernel_size, S, padding, dilation, ceil_mode)
    OH = calc_out_dim(H, kernel_size, S, padding, dilation, ceil_mode)
    OW = calc_out_dim(W, kernel_size, S, padding, dilation, ceil_mode)

    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

    # Input strides
    stride_n = C * D * H * W
    stride_c = D * H * W
    stride_d = H * W
    stride_h = W
    stride_w = 1

    # Output strides
    out_stride_n = C * OD * OH * OW
    out_stride_c = OD * OH * OW
    out_stride_d = OH * OW
    out_stride_h = OW
    out_stride_w = 1

    # Grid: one program per output element
    grid = (N, C, OD, OH, OW)

    maxpool3d_kernel[grid](
        x, out, 
        N, C, D, H, W, 
        OD, OH, OW, 
        kernel_size, S, padding, dilation, 
        stride_n, stride_c, stride_d, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 3D using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        # return_indices is not implemented in this Triton kernel as the original Model.forward doesn't use it

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using the Triton kernel.

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