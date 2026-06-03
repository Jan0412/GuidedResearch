import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    # Pointers to inputs and outputs
    x_ptr, w_ptr, out_ptr,
    # Output dimensions
    batch_size, out_channels, out_h, out_w,
    # Input dimensions
    in_channels, in_h, in_w,
    # Kernel dimensions
    kernel_h, kernel_w,
    # Stride and padding
    stride_h, stride_w, padding_h, padding_w,
    # Block sizes for tiling
    BLOCK_SIZE_M: tl.constexpr,  # Output tile size for channels
    BLOCK_SIZE_N: tl.constexpr,  # Output tile size for spatial
    BLOCK_SIZE_K: tl.constexpr,  # Input channel tile size
):
    # Get program IDs
    pid_m = tl.program_id(0)  # For output channels
    pid_n = tl.program_id(1)  # For spatial positions
    pid_b = tl.program_id(2)  # For batch
    
    # Calculate output channel range for this block
    out_c_start = pid_m * BLOCK_SIZE_M
    out_c_offsets = out_c_start + tl.arange(0, BLOCK_SIZE_M)
    out_c_mask = out_c_offsets < out_channels
    
    # Calculate output spatial position
    # pid_n encodes both h and w positions
    tile_h = BLOCK_SIZE_N // (out_w // 4 + 1)  # Approximate tiling
    if tile_h == 0:
        tile_h = 1
    out_h_idx = (pid_n // (out_w // tile_h + 1)) * tile_h
    out_w_idx = (pid_n % (out_w // tile_h + 1)) * tile_h
    
    # Create spatial offset arrays
    out_h_offsets = out_h_idx + tl.arange(0, tile_h)[:, None]
    out_w_offsets = out_w_idx + tl.arange(0, tile_h)[None, :]
    
    # Create masks for valid positions
    h_mask = out_h_offsets < out_h
    w_mask = out_w_offsets < out_w
    hw_mask = h_mask & w_mask
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, tile_h, tile_h), dtype=tl.float32)
    
    # Loop over input channels
    for k in range(0, in_channels, BLOCK_SIZE_K):
        in_c_start = k
        in_c_offsets = in_c_start + tl.arange(0, BLOCK_SIZE_K)
        in_c_mask = in_c_offsets < in_channels
        
        # Loop over kernel positions
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate corresponding input position
                in_h_pos = out_h_idx + kh - padding_h
                in_w_pos = out_w_idx + kw - padding_w
                
                # Check if input position is valid
                in_h_valid = (in_h_pos >= 0) & (in_h_pos < in_h)
                in_w_valid = (in_w_pos >= 0) & (in_w_pos < in_w)
                valid = in_h_valid & in_w_valid
                
                # Load input if valid
                if valid:
                    in_h_idx = in_h_pos
                    in_w_idx = in_w_pos
                    x_offset = ((pid_b * in_channels + in_c_offsets) * in_h * in_w + 
                               in_h_idx * in_w + in_w_idx)
                    x = tl.load(x_ptr + x_offset, mask=in_c_mask[:, None, None], other=0.0)
                else:
                    x = tl.zeros((BLOCK_SIZE_K, tile_h, tile_h), dtype=tl.float32)
                
                # Load kernel weight
                w_offset = ((out_c_start + tl.arange(0, BLOCK_SIZE_M)[:, None, None]) * kernel_h * kernel_w * in_channels +
                           kh * kernel_w * in_channels + kw * in_channels + 
                           (in_c_start + tl.arange(0, BLOCK_SIZE_K)[None, :, None]))
                w = tl.load(w_ptr + w_offset, mask=out_c_mask[:, None, None] & in_c_mask[None, :, None], other=0.0)
                
                # Accumulate: out += x * w
                acc += tl.sum(x[None, :, :] * w[:, :, :], axis=1)
    
    # Store result
    out_offset = ((pid_b * out_channels + out_c_offsets[:, None, None]) * out_h * out_w +
                 out_h_offsets[None, :, :] * out_w + out_w_offsets[None, :, :])
    tl.store(out_ptr + out_offset, acc, mask=out_c_mask[:, None, None] & hw_mask[None, :, :])


def triton_conv_transpose2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of ConvTranspose2d
    
    Args:
        x: Input tensor of shape (batch_size, in_channels, in_h, in_w)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_h, kernel_w)
        bias: Optional bias tensor
        stride, padding, output_padding, dilation, groups: Convolution parameters
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, in_h, in_w = x.shape
    _, out_channels, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    output_padding_h, output_padding_w = output_padding
    dilation_h, dilation_w = dilation
    
    # Calculate output shape
    out_h = (in_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1
    out_w = (in_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1
    
    # Prepare output tensor
    out = torch.zeros(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_M = 32  # Output channel block size
    BLOCK_SIZE_N = 64  # Spatial block size
    BLOCK_SIZE_K = 32  # Input channel block size
    
    # Grid dimensions
    grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = ((out_h * out_w) + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n, batch_size)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, out,
        batch_size, out_channels, out_h, out_w,
        in_channels, in_h, in_w,
        kernel_h, kernel_w,
        stride_h, stride_w, padding_h, padding_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Add bias if provided
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), output_padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
            
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, dilation=self.dilation,
            groups=self.groups
        )