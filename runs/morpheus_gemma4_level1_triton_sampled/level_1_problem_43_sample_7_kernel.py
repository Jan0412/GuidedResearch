import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def maxpool3d_kernel(
    x_ptr,
    out_ptr,
    S_N, S_C, S_D, S_H, S_W,
    OS_N, OS_C, OS_D, OS_H, OS_W,
    N, C, D, H, W,
    D_out, H_out, W_out,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Grid: (N * C * D_out, H_out, tl.cdiv(W_out, BLOCK_SIZE_W))
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid_2 = tl.program_id(2)

    # Decompose pid_0 into N, C, D_out
    n = pid_0 // (C * D_out)
    rem = pid_0 % (C * D_out)
    c = rem // D_out
    d_out = rem % D_out
    h_out = pid_1
    
    # Width offsets for the block
    w_out_offsets = pid_2 * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    mask_w = w_out_offsets < W_out

    # Initialize max values for the current block of output elements
    max_vals = tl.full([BLOCK_SIZE_W], -float('inf'), dtype=tl.float32)

    # Iterate over the 3D kernel window
    for kd in range(kernel_size):
        in_d = d_out * stride + kd * dilation - padding
        if in_d < 0 or in_d >= D:
            continue
            
        for kh in range(kernel_size):
            in_h = h_out * stride + kh * dilation - padding
            if in_h < 0 or in_h >= H:
                continue
                
            for kw in range(kernel_size):
                # Calculate input width indices for the block
                in_w = w_out_offsets * stride + kw * dilation - padding
                
                # Mask for bounds checking in the width dimension
                mask = mask_w & (in_w >= 0) & (in_w < W)
                
                # Compute pointer to the input elements
                # x_ptr + n*S_N + c*S_C + in_d*S_D + in_h*S_H + in_w*S_W
                input_ptrs = x_ptr + (n * S_N) + (c * S_C) + (in_d * S_D) + (in_h * S_H) + (in_w * S_W)
                
                # Load values and update max
                vals = tl.load(input_ptrs, mask=mask, other=-float('inf'))
                max_vals = tl.maximum(max_vals, vals)

    # Store the results back to the output tensor
    out_ptrs = out_ptr + (n * OS_N) + (c * OS_C) + (d_out * OS_D) + (h_out * OS_H) + (w_out_offsets * OS_W)
    tl.store(out_ptrs, max_vals, mask=mask_w)


def triton_maxpool3d(x, kernel_size, stride, padding, dilation, ceil_mode=False):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    N, C, D, H, W = x.shape
    
    if stride is None:
        stride = kernel_size

    # Calculate output dimensions
    def calc_out(size):
        out = (size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        return math.ceil(out) if ceil_mode else math.floor(out)

    D_out = calc_out(D)
    H_out = calc_out(H)
    W_out = calc_out(W)

    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    S_N, S_C, S_D, S_H, S_W = x.stride()
    OS_N, OS_C, OS_D, OS_H, OS_W = out.stride()

    BLOCK_SIZE_W = 32
    grid = (N * C * D_out, H_out, triton.cdiv(W_out, BLOCK_SIZE_W))

    maxpool3d_kernel[grid](
        x, out,
        S_N, S_C, S_D, S_H, S_W,
        OS_N, OS_C, OS_D, OS_H, OS_W,
        N, C, D, H, W,
        D_out, H_out, W_out,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
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
        # return_indices is not implemented in this Triton kernel as it's rarely used for speed-critical paths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using the Triton kernel.
        """
        return triton_maxpool3d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.ceil_mode
        )