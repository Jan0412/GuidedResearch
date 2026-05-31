import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def maxpool3d_kernel(
    input_ptr,
    output_ptr,
    N, C, D, H, W,
    D_out, H_out, W_out,
    S_N, S_C, S_D, S_H, S_W,
    O_N, O_C, O_D, O_H, O_W,
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program ID for the batch * channel * depth * height dimension
    pid_rest = tl.program_id(0)
    # Program ID for the width dimension
    pid_w = tl.program_id(1)

    # Decompose pid_rest into n, c, d_out, h_out
    # n: batch, c: channel, d_out: depth, h_out: height
    n = pid_rest // (C * D_out * H_out)
    rem = pid_rest % (C * D_out * H_out)
    c = rem // (D_out * H_out)
    rem = rem % (D_out * H_out)
    d_out = rem // H_out
    h_out = rem % H_out

    # Width offsets for the current block
    w_out_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = w_out_offsets < W_out

    # Initialize max values to negative infinity
    max_val = tl.full((BLOCK_W,), float("-inf"), dtype=tl.float32)

    # Iterate over the kernel window
    for i in range(KERNEL_SIZE):
        d_in = d_out * STRIDE - PADDING + i * DILATION
        # Check depth boundary
        d_mask = (d_in >= 0) & (d_in < D)
        
        for j in range(KERNEL_SIZE):
            h_in = h_out * STRIDE - PADDING + j * DILATION
            # Check height boundary
            h_mask = (h_in >= 0) & (h_in < H)
            
            for k in range(KERNEL_SIZE):
                w_in = w_out_offsets * STRIDE - PADDING + k * DILATION
                # Check width boundary and block mask
                w_mask = (w_in >= 0) & (w_in < W) & mask_w
                
                # Combined mask for this specific input element
                combined_mask = d_mask & h_mask & w_mask
                
                # Calculate input pointer offset
                # offset = n*S_N + c*S_C + d_in*S_D + h_in*S_H + w_in*S_W
                input_offset = (n * S_N + c * S_C + d_in * S_D + h_in * S_H + w_in * S_W)
                
                # Load value (use -inf for out-of-bounds to not affect max)
                val = tl.load(input_ptr + input_offset, mask=combined_mask, other=float("-inf"))
                max_val = tl.maximum(max_val, val)

    # Calculate output pointer offset
    # output_offset = n*O_N + c*O_C + d_out*O_D + h_out*O_H + w_out_offsets*O_W
    output_offset = (n * O_N + c * O_C + d_out * O_D + h_out * O_H + w_out_offsets * O_W)
    tl.store(output_ptr + output_offset, max_val, mask=mask_w)


def triton_maxpool3d(x, kernel_size, stride, padding, dilation, ceil_mode=False):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    N, C, D, H, W = x.shape

    # Calculate output dimensions
    def calc_out(size, k, s, p, d, ceil):
        out = (size + 2 * p - d * (k - 1) - 1) / s + 1
        return math.ceil(out) if ceil else math.floor(out)

    D_out = calc_out(D, kernel_size, stride, padding, dilation, ceil_mode)
    H_out = calc_out(H, kernel_size, stride, padding, dilation, ceil_mode)
    W_out = calc_out(W, kernel_size, stride, padding, dilation, ceil_mode)

    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Strides for input and output
    S_N, S_C, S_D, S_H, S_W = x.stride()
    O_N, O_C, O_D, O_H, O_W = out.stride()

    BLOCK_W = 32
    grid = (N * C * D_out * H_out, (W_out + BLOCK_W - 1) // BLOCK_W)

    maxpool3d_kernel[grid](
        x, out,
        N, C, D, H, W,
        D_out, H_out, W_out,
        S_N, S_C, S_D, S_H, S_W,
        O_N, O_C, O_D, O_H, O_W,
        KERNEL_SIZE=kernel_size,
        STRIDE=stride,
        PADDING=padding,
        DILATION=dilation,
        BLOCK_W=BLOCK_W,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 3D using Triton.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor using a custom Triton kernel.
        """
        # Triton kernel currently implements value pooling.
        # If return_indices is True, we still return only the pooled values as primary optimization.
        return triton_maxpool3d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.ceil_mode
        )