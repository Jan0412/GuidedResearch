import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def maxpool3d_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    oD, oH, oW,
    kernel_size, stride, padding, dilation,
    S_N, S_C, S_D, S_H, S_W,
    oS_N, oS_C, oS_D, oS_H, oS_W,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid mapping: (N * C, oD, oH, oW // BLOCK_SIZE)
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    pid3 = tl.program_id(3)

    b = pid0 // C
    c = pid0 % C
    od = pid1
    oh = pid2
    ow_offsets = pid3 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_ow = ow_offsets < oW

    # Initialize max_val with a very small number
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)

    # Loop over the kernel window
    for i in range(kernel_size):
        id_val = od * stride + i * dilation - padding
        if 0 <= id_val < D:
            for j in range(kernel_size):
                ih_val = oh * stride + j * dilation - padding
                if 0 <= ih_val < H:
                    for k in range(kernel_size):
                        iw_val = ow_offsets * stride + k * dilation - padding
                        mask_iw = (iw_val >= 0) & (iw_val < W) & mask_ow
                        
                        # Calculate input offset: x[b, c, id_val, ih_val, iw_val]
                        offset = b * S_N + c * S_C + id_val * S_D + ih_val * S_H + iw_val * S_W
                        val = tl.load(x_ptr + offset, mask=mask_iw, other=-float('inf'))
                        max_val = tl.maximum(max_val, val)

    # Calculate output offset: out[b, c, od, oh, ow]
    out_offset = b * oS_N + c * oS_C + od * oS_D + oh * oS_H + ow_offsets * oS_W
    tl.store(out_ptr + out_offset, max_val, mask=mask_ow)

def triton_maxpool3d(x, kernel_size, stride, padding, dilation, ceil_mode=False):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    N, C, D, H, W = x.shape
    
    if stride is None:
        stride = kernel_size

    def calc_out_dim(i, p, d, k, s, ceil):
        res = (i + 2 * p - d * (k - 1) - 1) / s + 1
        return math.ceil(res) if ceil else math.floor(res)

    oD = calc_out_dim(D, padding, dilation, kernel_size, stride, ceil_mode)
    oH = calc_out_dim(H, padding, dilation, kernel_size, stride, ceil_mode)
    oW = calc_out_dim(W, padding, dilation, kernel_size, stride, ceil_mode)

    out = torch.empty((N, C, oD, oH, oW), device=x.device, dtype=x.dtype)

    # Strides for input
    S_W = 1
    S_H = W
    S_D = H * W
    S_C = D * H * W
    S_N = C * D * H * W

    # Strides for output
    oS_W = 1
    oS_H = oW
    oS_D = oH * oW
    oS_C = oD * oH * oW
    oS_N = C * oD * oH * oW

    BLOCK_SIZE = 128
    grid = (N * C, oD, oH, triton.cdiv(oW, BLOCK_SIZE))

    maxpool3d_kernel[grid](
        x, out,
        N, C, D, H, W,
        oD, oH, oW,
        kernel_size, stride, padding, dilation,
        S_N, S_C, S_D, S_H, S_W,
        oS_N, oS_C, oS_D, oS_H, oS_W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 3D using custom Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        # return_indices is not implemented in this Triton kernel for simplicity
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using the custom Triton kernel.
        """
        # Triton kernel implementation for MaxPool3d
        return triton_maxpool3d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.ceil_mode
        )