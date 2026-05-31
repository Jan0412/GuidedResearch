import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    bias_ptr,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    batch_size,
    depth,
    height,
    width,
    depth_out,
    height_out,
    width_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid mapping
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for M (batch * out_channels) and N (output spatial)
    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N

    # Create masks for M and N dimensions
    off_m = tl.arange(0, BLOCK_M) + m_offset
    off_n = tl.arange(0, BLOCK_N) + n_offset
    mask_m = off_m < batch_size * out_channels
    mask_n = off_n < depth_out * height_out * width_out

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K dimension is in_channels * kernel_size^3
    k_dim = in_channels * kernel_size * kernel_size * kernel_size

    # Loop over K dimension in blocks
    for k in range(0, k_dim, BLOCK_K):
        off_k = tl.arange(0, BLOCK_K) + k
        mask_k = off_k < k_dim

        # Load weight tile: shape (BLOCK_M, BLOCK_K)
        # Weight layout: (out_channels, in_channels * K^3)
        # We need to handle batch dimension in M offset
        w_ptr_off = w_ptr + off_m[:, None] * k_dim + off_k[None, :]
        w_tile = tl.load(w_ptr_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load input tile: shape (BLOCK_K, BLOCK_N)
        # Input layout: (batch, in_channels, depth, height, width)
        # We need to gather patches for each output location n
        # Output location n corresponds to (b, d, h, w)
        # d_in = (d - kd*dilation + padding) // stride
        # h_in = (h - kh*dilation + padding) // stride
        # w_in = (w - kw*dilation + padding) // stride
        
        # Compute output coordinates for n
        # n = d * (height_out * width_out) + h * width_out + w
        n_coords = off_n[None, :]
        w_coords = n_coords % width_out
        h_coords = (n_coords // width_out) % height_out
        d_coords = n_coords // (height_out * width_out)
        
        # Batch index is part of m_offset
        # m_offset = b * out_channels + co
        # We need b for x access
        b_coords = off_m[:, None] // out_channels
        
        # Generate kernel coordinates kd, kh, kw
        # k_dim = C_in * K^3
        # k = ci * K^3 + kd * K^2 + kh * K + kw
        # We need to map off_k to ci, kd, kh, kw
        # off_k is the index in the flattened K dimension
        # ci = off_k // (K^3)
        # rem = off_k % (K^3)
        # kd = rem // (K^2)
        # kh = rem % (K^2) // K
        # kw = rem % K
        
        K2 = kernel_size * kernel_size
        K3 = K2 * kernel_size
        
        ci = off_k[None, :] // K3
        rem = off_k[None, :] % K3
        kd = rem // K2
        kh = rem % K2 // kernel_size
        kw = rem % kernel_size
        
        # Compute input coordinates
        # d_in = (d - kd*dilation + padding) // stride
        # We need to compute this for each (b, co, d, h, w, ci, kd, kh, kw)
        # Broadcasting:
        # b_coords: (BLOCK_M, 1)
        # d_coords: (1, BLOCK_N)
        # kd: (1, BLOCK_K)
        # Similarly for h, w, kh, kw
        
        d_in = (d_coords - kd * dilation + padding) // stride
        h_in = (h_coords - kh * dilation + padding) // stride
        w_in = (w_coords - kw * dilation + padding) // stride
        
        # Create masks for valid input coordinates
        mask_d = (d_in >= 0) & (d_in < depth)
        mask_h = (h_in >= 0) & (h_in < height)
        mask_w = (w_in >= 0) & (w_in < width)
        mask_x = mask_d & mask_h & mask_w
        
        # Compute x pointer offsets
        # x layout: (batch, in_channels, depth, height, width)
        # stride_x = (height * width, width, 1)
        # offset = b * C_in * D * H * W + ci * D * H * W + d_in * H * W + h_in * W + w_in
        
        # Precompute strides
        stride_d = height * width
        stride_h = width
        stride_b = in_channels * depth * height * width
        
        x_ptr_off = (b_coords * stride_b) + (ci * stride_d) + (d_in * stride_h) + h_in + w_in
        x_tile = tl.load(x_ptr_off, mask=mask_x & mask_k[None, :], other=0.0)

        # Dot product accumulation
        acc += tl.dot(w_tile, x_tile)

    # Reshape acc to (BLOCK_M, BLOCK_N) -> (B, C_out, D_out, H_out, W_out)
    # Store to output tensor
    # out layout: (batch, out_channels, depth_out, height_out, width_out)
    # We need to map (m, n) back to (b, co, d, h, w)
    # m = b * out_channels + co
    # n = d * H_out * W_out + h * W_out + w
    
    # Compute output coordinates from m and n
    co = off_m[:, None] % out_channels
    b_out = off_m[:, None] // out_channels
    
    # n_coords already computed
    # w_out_coords, h_out_coords, d_out_coords
    
    # Compute out pointer offsets
    # out_ptr offset = b * C_out * D_out * H_out * W_out + co * D_out * H_out * W_out + d * H_out * W_out + h * W_out + w
    
    stride_d_out = height_out * width_out
    stride_h_out = width_out
    stride_b_out = out_channels * depth_out * height_out * width_out
    
    out_ptr_off = (b_out * stride_b_out) + (co * stride_d_out) + (d_coords * stride_h_out) + h_coords + w_coords
    
    # Add bias if available
    if bias_ptr is not None:
        # Bias shape: (out_channels,)
        # Bias offset = co
        bias_off = co
        bias = tl.load(bias_ptr + bias_off, mask=mask_m[:, None], other=0.0)
        acc += bias

    # Store result
    tl.store(out_ptr_off, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_conv_transpose3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int, kernel_size: int):
    """
    Wrapper function for the custom Triton ConvTranspose3d kernel.
    """
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, k_h, k_w, k_d = weight.shape
    assert k_h == k_w == k_d == kernel_size

    # Compute output dimensions
    depth_out = (depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + 1
    height_out = (height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + 1
    width_out = (width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + 1

    out = torch.empty((batch_size, out_channels, depth_out, height_out, width_out), dtype=x.dtype, device=x.device)

    # Determine block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 64

    # Grid calculation
    num_m_blocks = (batch_size * out_channels + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (depth_out * height_out * width_out + BLOCK_N - 1) // BLOCK_N

    grid = (num_m_blocks, num_n_blocks)

    conv_transpose3d_kernel[grid](
        x, weight, out, bias,
        in_channels, out_channels, kernel_size, stride, padding, dilation,
        batch_size, depth, height, width,
        depth_out, height_out, width_out,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias if self.conv_transpose3d.bias is not None else None
        return triton_conv_transpose3d(x, weight, bias, self.stride, self.padding, self.dilation, self.kernel_size)