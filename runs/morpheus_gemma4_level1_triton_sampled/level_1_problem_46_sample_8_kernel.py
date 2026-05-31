import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool3d_kernel(
    x_ptr, 
    out_ptr,
    N, C, D, H, W,
    D_out, H_out, W_out,
    k, s, p,
    S_N, S_C, S_D, S_H, S_W,
    OS_N, OS_C, OS_D, OS_H, OS_W,
    BLOCK_W: tl.constexpr,
):
    # Each program computes a block of output width elements for a specific (n, c, d, h)
    pid_rest = tl.program_id(0)
    pid_w = tl.program_id(1)

    # Decode pid_rest into n, c, d, h
    h = pid_rest % H_out
    pid_rest //= H_out
    d = pid_rest % D_out
    pid_rest //= D_out
    c = pid_rest % C
    pid_rest //= C
    n = pid_rest

    # Output width offsets for this block
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W_out

    # Accumulator for the average pooling sum
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Iterate over the kernel window (k x k x k)
    for dz in range(k):
        z_in = d * s + dz - p
        if z_in < 0 or z_in >= D:
            continue
        for dy in range(k):
            y_in = h * s + dy - p
            if y_in < 0 or y_in >= H:
                continue
            for dx in range(k):
                # Calculate input indices for the width dimension
                x_in_offsets = w_offsets * s + dx - p
                mask_x = (x_in_offsets >= 0) & (x_in_offsets < W) & mask_w
                
                # Load values from the input tensor
                # Pointer arithmetic: base + n*S_N + c*S_C + z_in*S_D + y_in*S_H + x_in_offsets*S_W
                ptr = x_ptr + n * S_N + c * S_C + z_in * S_D + y_in * S_H + x_in_offsets * S_W
                acc += tl.load(ptr, mask=mask_x, other=0.0)

    # Compute average: divide by the total volume of the kernel window
    acc = acc / (k * k * k)

    # Store the result in the output tensor
    out_ptr_base = out_ptr + n * OS_N + c * OS_C + d * OS_D + h * OS_H
    tl.store(out_ptr_base + w_offsets * OS_W, acc, mask=mask_w)


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    # Ensure input is contiguous on GPU
    x = x.contiguous().cuda()
    N, C, D, H, W = x.shape
    
    # Calculate output dimensions
    D_out = (D + 2 * padding - kernel_size) // stride + 1
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Input strides
    S_N = C * D * H * W
    S_C = D * H * W
    S_D = H * W
    S_H = W
    S_W = 1
    
    # Output strides
    OS_N = C * D_out * H_out * W_out
    OS_C = D_out * H_out * W_out
    OS_D = H_out * W_out
    OS_H = W_out
    OS_W = 1
    
    BLOCK_W = 16
    # Grid: parallelize over (N, C, D_out, H_out) and then over W_out in blocks
    grid = (N * C * D_out * H_out, triton.cdiv(W_out, BLOCK_W))
    
    avg_pool3d_kernel[grid](
        x, out,
        N, C, D, H, W,
        D_out, H_out, W_out,
        kernel_size, stride, padding,
        S_N, S_C, S_D, S_H, S_W,
        OS_N, OS_C, OS_D, OS_H, OS_W,
        BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)