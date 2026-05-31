import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool_3d_kernel(
    x_ptr,
    out_ptr,
    B, C, D, H, W,
    out_d, out_h, out_w,
    stride,
    padding,
    S_B, S_C, S_D, S_H, S_W,
    OS_B, OS_C, OS_D, OS_H, OS_W,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for the output plane (Batch, Channel, Depth, Height)
    pid_out_plane = tl.program_id(0)
    # Program ID for the width block
    pid_w_block = tl.program_id(1)

    # Decompose pid_out_plane into b, c, d, h
    # b = pid_out_plane // (C * out_d * out_h)
    # rem = pid_out_plane % (C * out_d * out_h)
    # c = rem // (out_d * out_h)
    # rem = rem % (out_d * out_h)
    # d = rem // out_h
    # h = rem % out_h
    
    # Optimization: use precomputed divisors
    C_S = C * out_d * out_h
    D_S = out_d * out_h
    H_S = out_h
    
    b = pid_out_plane // C_S
    rem = pid_out_plane % C_S
    c = rem // D_S
    rem = rem % D_S
    d = rem // H_S
    h = rem % H_S

    # Width offsets for this block
    w_offsets = pid_w_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_w = w_offsets < out_w

    # Accumulator for the sum
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the 3D kernel window
    for i in range(KERNEL_SIZE):
        z = d * stride + i - padding
        mask_z = (z >= 0) & (z < D)
        
        for j in range(KERNEL_SIZE):
            y = h * stride + j - padding
            mask_y = (y >= 0) & (y < H)
            
            for k in range(KERNEL_SIZE):
                x_coords = w_offsets * stride + k - padding
                mask_x = (x_coords >= 0) & (x_coords < W)
                
                # Final mask for loading
                # Note: mask_z and mask_y are scalars for the block, mask_x and mask_w are vectors
                load_mask = mask_z & mask_y & mask_x & mask_w
                
                # Calculate input pointer
                # ptr = x_ptr + b*S_B + c*S_C + z*S_D + y*S_H + x_coords*S_W
                ptr = x_ptr + b * S_B + c * S_C + z * S_D + y * S_H + x_coords * S_W
                
                # Load and accumulate
                val = tl.load(ptr, mask=load_mask, other=0.0)
                sum_val += val

    # Calculate average
    out_val = sum_val / (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)

    # Store the result
    # out_ptr + b*OS_B + c*OS_C + d*OS_D + h*OS_H + w_offsets*OS_W
    out_ptr_final = out_ptr + b * OS_B + c * OS_C + d * OS_D + h * OS_H + w_offsets * OS_W
    tl.store(out_ptr_final, out_val, mask=mask_w)


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    B, C, D, H, W = x.shape
    if stride is None:
        stride = kernel_size

    # Calculate output dimensions
    out_d = (D + 2 * padding - kernel_size) // stride + 1
    out_h = (H + 2 * padding - kernel_size) // stride + 1
    out_w = (W + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty((B, C, out_d, out_h, out_w), device=x.device, dtype=x.dtype)

    # Get strides
    S_B, S_C, S_D, S_H, S_W = x.stride()
    OS_B, OS_C, OS_D, OS_H, OS_W = out.stride()

    BLOCK_SIZE = 64
    # Grid: (B * C * out_d * out_h, (out_w + BLOCK_SIZE - 1) // BLOCK_SIZE)
    grid = (B * C * out_d * out_h, (out_w + BLOCK_SIZE - 1) // BLOCK_SIZE)

    avg_pool_3d_kernel[grid](
        x, out,
        B, C, D, H, W,
        out_d, out_h, out_w,
        stride, padding,
        S_B, S_C, S_D, S_H, S_W,
        OS_B, OS_C, OS_D, OS_H, OS_W,
        KERNEL_SIZE=kernel_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 3D Average Pooling using a custom Triton kernel.
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
        # Ensure input is on GPU and contiguous for the kernel
        if not x.is_cuda:
            raise RuntimeError("Input tensor must be on CUDA.")
        
        # Using the wrapper function
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)