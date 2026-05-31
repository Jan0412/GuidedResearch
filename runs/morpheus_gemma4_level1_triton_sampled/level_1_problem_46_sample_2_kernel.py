import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool3d_kernel(
    x_ptr, 
    out_ptr,
    B, C, D, H, W,
    stride, padding,
    D_out, H_out, W_out,
    S_d, S_h, S_w, S_c,
    OS_d, OS_h, OS_w, OS_c,
    BLOCK_SIZE_W: tl.constexpr,
    K_SIZE: tl.constexpr,
):
    # Grid: (B * C, D_out, H_out, (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
    pid_bc = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w_block = tl.program_id(3)

    # Width offsets for the output
    w_offsets = pid_w_block * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    mask_w = w_offsets < W_out

    # Accumulator for the average pooling window
    acc = tl.zeros([BLOCK_SIZE_W], dtype=tl.float32)

    # Base pointer for the current batch and channel
    # Since we use pid_bc = b * C + c, and the tensor is contiguous,
    # the offset is pid_bc * S_c
    bc_ptr = x_ptr + pid_bc * S_c

    # Loop over the 3D kernel window
    for i in range(K_SIZE):
        d = pid_d * stride - padding + i
        if d < 0 or d >= D:
            continue
        # Pointer to the current depth slice
        d_ptr = bc_ptr + d * S_d

        for j in range(K_SIZE):
            h = pid_h * stride - padding + j
            if h < 0 or h >= H:
                continue
            # Pointer to the current height line
            h_ptr = d_ptr + h * S_h

            for k in range(K_SIZE):
                # Calculate input width offsets for this window element
                w_offsets_in = w_offsets * stride - padding + k
                # Mask to ensure we are within input boundaries and output boundaries
                mask_w_in = mask_w & (w_offsets_in >= 0) & (w_offsets_in < W)
                # Load values and add to accumulator
                vals = tl.load(h_ptr + w_offsets_in * S_w, mask=mask_w_in, other=0.0)
                acc += vals

    # Average pooling: divide by the total volume of the kernel
    acc = acc / (K_SIZE * K_SIZE * K_SIZE)

    # Calculate output pointer and store results
    out_base = out_ptr + pid_bc * OS_c + pid_d * OS_d + pid_h * OS_h
    tl.store(out_base + w_offsets * OS_w, acc, mask=mask_w)


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    B, C, D, H, W = x.shape

    # Handle default stride
    if stride is None:
        stride = kernel_size

    # Calculate output dimensions
    D_out = (D + 2 * padding - kernel_size) // stride + 1
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    # Prepare output tensor
    out = torch.empty((B, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Input strides
    S_w = x.stride(4)
    S_h = x.stride(3)
    S_d = x.stride(2)
    S_c = x.stride(1)

    # Output strides
    OS_w = out.stride(4)
    OS_h = out.stride(3)
    OS_d = out.stride(2)
    OS_c = out.stride(1)

    BLOCK_SIZE_W = 32
    grid = (B * C, D_out, H_out, (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

    avg_pool3d_kernel[grid](
        x, out,
        B, C, D, H, W,
        stride, padding,
        D_out, H_out, W_out,
        S_d, S_h, S_w, S_c,
        OS_d, OS_h, OS_w, OS_c,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        K_SIZE=kernel_size
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using the Triton implementation.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)