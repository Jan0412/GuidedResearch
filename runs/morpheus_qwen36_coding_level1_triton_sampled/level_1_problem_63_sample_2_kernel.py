import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W, C_out, K, H_out, W_out,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid mapping: 1D grid of size N * C_out * num_blocks
    pid = tl.program_id(0)
    num_blocks = (H_out * W_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    n = pid // (C_out * num_blocks)
    rest = pid % (C_out * num_blocks)
    c_out = rest // num_blocks
    block_idx = rest % num_blocks

    # Output block offsets
    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (H_out * W_out)

    h_out = offsets // W_out
    w_out = offsets % W_out

    # Load weights for current output channel
    # w_ptr shape: (C_out, C_in, K, K)
    w_base = w_ptr + c_out * C_in * K * K
    W_tile = tl.load(w_base + tl.arange(0, C_in * K * K), mask=tl.arange(0, C_in * K * K) < C_in * K * K, other=0.0)
    W_tile = tl.reshape(W_tile, (C_in, K, K))

    # Load bias
    b = tl.load(b_ptr + c_out)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Strides for input tensor
    stride_x_n = C_in * H * W
    stride_x_c = H * W
    stride_x_h = W
    stride_x_w = 1

    # Accumulate over input channels and kernel spatial dimensions
    for c_in in range(C_in):
        for k_h in range(K):
            for k_w in range(K):
                # Compute input coordinates
                h_in = h_out * stride - padding + k_h * dilation
                w_in = w_out * stride - padding + k_w * dilation

                # Compute linear offset for input
                offset_x = n * stride_x_n + c_in * stride_x_c + h_in * stride_x_h + w_in * stride_x_w

                # Load input with bounds checking
                load_mask = mask & (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                x_val = tl.load(x_ptr + offset_x, mask=load_mask, other=0.0)

                # Multiply by weight and accumulate
                w_val = W_tile[c_in, k_h, k_w]
                acc += x_val * w_val

    # Add bias
    acc += b

    # Store output
    # out_ptr shape: (N, C_out, H_out, W_out)
    stride_out_n = C_out * H_out * W_out
    stride_out_c = H_out * W_out
    stride_out_h = W_out
    stride_out_w = 1
    offset_out = n * stride_out_n + c_out * stride_out_c + h_out * stride_out_h + w_out * stride_out_w
    tl.store(out_ptr + offset_out, acc, mask=mask)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, stride: int, padding: int, dilation: int, groups: int) -> torch.Tensor:
    """
    Wrapper function to launch the custom Triton Conv2d kernel.
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    assert x.dim() == 4, "Input must be 4D (N, C_in, H, W)."
    assert w.dim() == 4, "Weights must be 4D (C_out, C_in, K, K)."
    
    N, C_in, H, W = x.shape
    C_out, _, K, _ = w.shape
    
    # Calculate output spatial dimensions
    H_out = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    
    # Ensure contiguous
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    
    # Prepare output
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Block size tuning
    BLOCK_SIZE = 128
    
    # Grid calculation
    num_blocks = (H_out * W_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (N * C_out * num_blocks,)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, w, b, out,
        N, C_in, H, W, C_out, K, H_out, W_out,
        stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # Ensure parameters are ready for Triton
        self.register_buffer('weight', self.conv2d.weight)
        if bias:
            self.register_buffer('bias', self.conv2d.bias)
        else:
            self.register_buffer('bias', torch.zeros(out_channels, dtype=torch.float32))
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)