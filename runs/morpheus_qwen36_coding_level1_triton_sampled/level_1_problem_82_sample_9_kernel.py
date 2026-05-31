import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    stride, padding, kernel_size,
    in_channels, height, width,
    out_height, out_width,
    BLOCK_SIZE: tl.constexpr
):
    # Grid coordinates
    b = tl.program_id(0)
    c = tl.program_id(1)
    ty = tl.program_id(2)
    tx = tl.program_id(3)

    # Output tile start coordinates
    y_h_start = ty * BLOCK_SIZE
    y_w_start = tx * BLOCK_SIZE

    # Input tile start coordinates based on stride and padding
    x_h_start = y_h_start * stride - padding
    x_w_start = y_w_start * stride - padding

    # Shared memory dimensions
    SHM_H = BLOCK_SIZE + kernel_size - 1
    SHM_W = BLOCK_SIZE + kernel_size - 1

    # Allocate shared memory for input patch
    # Initialize with zeros to handle boundary conditions
    shm = tl.static(tl.zeros((SHM_H, SHM_W), dtype=tl.float32))

    # Load input patch into shared memory
    # Loop over shared memory coordinates
    for sh in tl.range(SHM_H):
        for sw in tl.range(SHM_W):
            # Global input coordinates
            h_idx = x_h_start + sh
            w_idx = x_w_start + sw

            # Mask for valid access
            mask = (h_idx >= 0) & (h_idx < height) & (w_idx >= 0) & (w_idx < width)

            # Calculate linear index for input tensor
            # Input shape: (B, C, H, W)
            x_offset = b * in_channels * height * width + c * height * width + h_idx * width + w_idx

            # Load from global memory
            val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

            # Store to shared memory
            tl.store(shm + sh * SHM_W + sw, val, mask=mask)

    # Accumulator for the output tile
    acc = 0.0

    # Weight offset for the current channel
    # Weights shape: (in_channels, 1, kernel_size, kernel_size)
    w_offset = c * kernel_size * kernel_size

    # Compute convolution
    # Loop over output tile coordinates
    for y in tl.range(BLOCK_SIZE):
        for x in tl.range(BLOCK_SIZE):
            # Global output coordinates
            gy = y_h_start + y
            gx = y_w_start + x

            # Mask for valid output access
            out_mask = (gy < out_height) & (gx < out_width)

            if out_mask:
                # Accumulate result
                # Start accumulation
                tile_acc = 0.0

                # Loop over kernel
                for k_h in tl.range(kernel_size):
                    for k_w in tl.range(kernel_size):
                        # Load from shared memory
                        # Shared memory index
                        shm_idx = (y + k_h) * SHM_W + (x + k_w)
                        x_val = tl.load(shm + shm_idx)

                        # Load weight
                        w_idx = w_offset + k_h * kernel_size + k_w
                        w_val = tl.load(w_ptr + w_idx)

                        tile_acc += x_val * w_val

                acc = tile_acc

                # Add bias
                if bias_ptr is not None:
                    acc += tl.load(bias_ptr + c)

                # Store to output
                # Output shape: (B, C, H_out, W_out)
                out_offset = b * in_channels * out_height * out_width + c * out_height * out_width + gy * out_width + gx
                tl.store(out_ptr + out_offset, acc)


def triton_depthwise_conv(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor,
                          stride: int, padding: int, kernel_size: int,
                          in_channels: int, height: int, width: int) -> torch.Tensor:
    """
    Wrapper function to launch the Triton depthwise convolution kernel.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Calculate output dimensions
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    # Prepare output tensor
    batch_size = x.shape[0]
    out = torch.empty((batch_size, in_channels, out_height, out_width), dtype=x.dtype, device=x.device)

    # Block size tuning
    BLOCK_SIZE = 64

    # Grid calculation
    num_blocks_h = (out_height + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_w = (out_width + BLOCK_SIZE - 1) // BLOCK_SIZE

    grid = (batch_size, in_channels, num_blocks_h, num_blocks_w)

    # Launch kernel
    depthwise_conv_kernel[grid](
        x, w, bias, out,
        stride, padding, kernel_size,
        in_channels, height, width,
        out_height, out_width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            self.stride,
            self.padding,
            self.kernel_size,
            x.shape[1],
            x.shape[2],
            x.shape[3]
        )