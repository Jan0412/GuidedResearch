import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool_kernel(
    x_ptr, 
    out_ptr, 
    B, C, H, W, 
    H_out, W_out, 
    S, P, 
    stride_b, stride_c, stride_h, stride_w, 
    out_stride_b, out_stride_c, out_stride_h, out_stride_w, 
    K: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
):
    # Program IDs
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # Decompose pid_0 into batch, channel, and output height
    # pid_0 = b * (C * H_out) + c * H_out + h_out
    b = pid_0 // (C * H_out)
    rem = pid_0 % (C * H_out)
    c = rem // H_out
    h_out = rem % H_out

    # Map pid_1 to a block of output widths
    w_out_offsets = pid_1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_w_out = w_out_offsets < W_out

    # Initialize sum for the window
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the pooling kernel window
    for kh in range(K):
        h_in = h_out * S + kh - P
        if h_in >= 0 and h_in < H:
            # Compute base pointer for the current row in the input window
            row_ptr = x_ptr + b * stride_b + c * stride_c + h_in * stride_h
            for kw in range(K):
                # Compute window offsets for the current column
                w_in_offsets = w_out_offsets * S + kw - P
                mask_w_in = mask_w_out & (w_in_offsets >= 0) & (w_in_offsets < W)
                
                # Load and accumulate values
                vals = tl.load(row_ptr + w_in_offsets * stride_w, mask=mask_w_in, other=0.0)
                acc += vals

    # Compute average (divisor is K*K for count_include_pad=True)
    out = acc / (K * K)

    # Store the result in the output tensor
    out_ptr_base = out_ptr + b * out_stride_b + c * out_stride_c + h_out * out_stride_h
    tl.store(out_ptr_base + w_out_offsets * out_stride_w, out, mask=mask_w_out)


def triton_avg_pool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    # Ensure input is contiguous on GPU
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    B, C, H, W = x.shape
    S = stride if stride is not None else kernel_size
    P = padding
    K = kernel_size

    # Calculate output dimensions
    H_out = (H + 2 * P - K) // S + 1
    W_out = (W + 2 * P - K) // S + 1

    # Prepare output tensor
    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Get strides for efficient pointer arithmetic
    stride_b, stride_c, stride_h, stride_w = x.stride()
    out_stride_b, out_stride_c, out_stride_h, out_stride_w = out.stride()

    # Define block size and grid
    BLOCK_SIZE = 128
    grid = (B * C * H_out, triton.cdiv(W_out, BLOCK_SIZE))

    # Launch kernel
    avg_pool_kernel[grid](
        x, out, 
        B, C, H, W, 
        H_out, W_out, 
        S, P, 
        stride_b, stride_c, stride_h, stride_w, 
        out_stride_b, out_stride_c, out_stride_h, out_stride_w, 
        K=K, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using Triton.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)