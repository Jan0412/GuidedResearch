import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    stride,
    padding,
    kernel_size,
    has_bias,
    in_channels,
    out_channels,
    height_out,
    width_out,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid coordinates
    pid = tl.program_id(0)
    num_width_blocks = tl.cdiv(width_out, BLOCK_SIZE)
    
    # Decode batch, out_channel, height, and width block
    b = pid // (out_channels * height_out * num_width_blocks)
    c_out = (pid // (height_out * num_width_blocks)) % out_channels
    h = (pid // num_width_blocks) % height_out
    w_block = pid % num_width_blocks
    
    # Width offsets for the block
    w_start = w_block * BLOCK_SIZE
    offsets_w = w_start + tl.arange(0, BLOCK_SIZE)
    mask_w = offsets_w < width_out
    
    # Load weights for the current output channel into shared memory
    # Weight shape: (out_channels, in_channels, kernel_size, kernel_size)
    # We need slice [c_out, :, :, :]
    w_offset = c_out * in_channels * kernel_size * kernel_size
    w_ptr_slice = w_ptr + w_offset
    
    # Shared memory for weights
    w_shared = tl.static_assert(kernel_size * kernel_size * in_channels <= 4096, "Weight slice too large for shared memory")
    w_shared_mem = tl.make_block_descriptor(w_ptr_slice, (in_channels * kernel_size * kernel_size,), BLOCK_SIZE, BLOCK_SIZE)
    # Since we load the whole slice for the block, we can use a single load if we handle masking
    # However, Triton shared memory loading is easier with tl.load on pointers
    # We'll load into a 1D array in shared memory conceptually, but in Triton we just load into registers or use tl.load with offsets
    # For simplicity and correctness, we load the weights into a shared memory array
    # Size: in_channels * kernel_size * kernel_size
    w_size = in_channels * kernel_size * kernel_size
    w_shared_arr = tl.arange(0, w_size)
    mask_w_arr = w_shared_arr < w_size
    
    # Load weights
    # Note: w_ptr_slice points to the start of the slice
    # We load w_size elements
    w_vals = tl.load(w_ptr_slice + w_shared_arr, mask=mask_w_arr, other=0.0)
    
    # Accumulator for each thread
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over in_channels, kernel_h, kernel_w
    # We can fuse these loops
    for c_in in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute input coordinates
                x_h = h * stride + kh - padding
                x_w = offsets_w * stride + kw - padding
                
                # Check bounds
                mask_x_h = (x_h >= 0) & (x_h < height)
                mask_x_w = (x_w >= 0) & (x_w < width)
                mask_x = mask_x_h[:, None] & mask_x_w[None, :] # This is wrong for vectorized access
                
                # Correct mask for x_w: offsets_w is vectorized
                # x_w is vectorized, x_h is scalar
                mask_x = (x_h >= 0) & (x_h < height) & (x_w >= 0) & (x_w < width)
                
                # Load x
                # x shape: (batch, in_channels, height, width)
                # offset = b * in_channels * height * width + c_in * height * width + x_h * width + x_w
                x_offset = b * in_channels * height * width + c_in * height * width + x_h * width + x_w
                x_vals = tl.load(x_ptr + x_offset, mask=mask_x, other=0.0)
                
                # Load weight
                w_idx = c_in * kernel_size * kernel_size + kh * kernel_size + kw
                w_val = w_vals[w_idx]
                
                acc += x_vals * w_val
    
    # Add bias
    if has_bias:
        acc += tl.load(b_ptr + c_out)
    
    # Store output
    # out shape: (batch, out_channels, height_out, width_out)
    out_offset = b * out_channels * height_out * width_out + c_out * height_out * width_out + h * width_out + offsets_w
    tl.store(out_ptr + out_offset, acc, mask=mask_w)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    height_out = (height - 1) * stride + kernel_size - 2 * padding + output_padding
    width_out = (width - 1) * stride + kernel_size - 2 * padding + output_padding
    
    out = torch.empty(batch_size, out_channels, height_out, width_out, device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 16
    num_width_blocks = triton.cdiv(width_out, BLOCK_SIZE)
    grid = (batch_size * out_channels * height_out * num_width_blocks,)
    
    has_bias = 1 if bias is not None else 0
    
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        stride, padding, kernel_size, has_bias,
        in_channels, out_channels, height_out, width_out,
        height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias
        
        # Initialize weights and bias
        # Using Kaiming uniform for weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='linear')
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding, self.groups
        )