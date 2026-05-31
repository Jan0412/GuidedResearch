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
    B, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K_h, K_w,
    stride_h, stride_w,
    padding_h, padding_w,
    output_padding_h, output_padding_w,
    dilation_h, dilation_w,
    groups,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Decode program ID to output coordinates
    pid = tl.program_id(0)
    
    # Calculate batch, output channel, and spatial coordinates
    num_elements = B * C_out * H_out * W_out
    if pid >= num_elements:
        return
        
    ow = pid % W_out
    pid //= W_out
    oh = pid % H_out
    pid //= H_out
    c_out = pid % C_out
    b = pid // C_out
    
    # Determine group for this output channel
    group_id = c_out // (C_out // groups)
    c_in_start = group_id * (C_in // groups)
    c_in_end = (group_id + 1) * (C_in // groups)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in tl.arange(0, BLOCK_SIZE_K):
        kh_valid = kh < K_h
        for kw in tl.arange(0, BLOCK_SIZE_K):
            kw_valid = kw < K_w
            
            # Compute input coordinates for this kernel position
            ih = oh * stride_h - padding_h + kh * dilation_h
            iw = ow * stride_w - padding_w + kw * dilation_w
            
            # Mask for valid input coordinates
            ih_valid = ih >= 0 & ih < H_in
            iw_valid = iw >= 0 & iw < W_in
            k_valid = kh_valid & kw_valid
            
            # Load input tile for current kernel position
            # We load a block of input channels
            c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C)
            c_in_mask = (c_in_offsets >= c_in_start) & (c_in_offsets < C_in)
            
            # Input pointer calculation
            # x[b, c_in, ih, iw]
            x_offsets = b * C_in * H_in * W_in + c_in_offsets * H_in * W_in + ih * W_in + iw
            x_mask = c_in_mask & ih_valid & iw_valid
            
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            
            # Load weight tile for current kernel position
            # w[c_out, c_in, kh, kw]
            w_offsets = c_out * C_in * K_h * K_w + c_in_offsets * K_h * K_w + kh * K_w + kw
            w_mask = c_in_mask
            
            w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)
            
            # Multiply and accumulate
            acc += tl.sum(x_vals * w_vals, axis=0) * k_valid
            
            # Note: In a real optimized kernel, we would fuse loops better
            # and handle masking more efficiently, but this is a functional baseline.
            # For better performance, tiling over K and C_in simultaneously
            # with proper masking would be required.
            
    # Add bias
    if b_ptr is not None:
        acc += tl.load(b_ptr + c_out)
        
    # Store result
    out_offset = b * C_out * H_out * W_out + c_out * H_out * W_out + oh * W_out + ow
    tl.store(out_ptr + out_offset, acc)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                            stride: tuple = (1, 1), padding: tuple = (0, 0),
                            output_padding: tuple = (0, 0), dilation: tuple = (1, 1),
                            groups: int = 1) -> torch.Tensor:
    """
    Wrapper function for the Triton ConvTranspose2d kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
        
    B, C_in, H_in, W_in = x.shape
    C_out, _, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_h - 1) + output_padding[0] + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_w - 1) + output_padding[1] + 1
    
    out = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Block sizes for tiling
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_K = 4
    
    # Grid size: one block per output element
    n_elements = B * C_out * H_out * W_out
    grid = (n_elements,)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride[0], stride[1],
        padding[0], padding[1],
        output_padding[0], output_padding[1],
        dilation[0], dilation[1],
        groups,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized transposed 2D convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias if self.conv_transpose2d.bias is not None else None
        return triton_conv_transpose2d(
            x, weight, bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )


def get_inputs():
    # randomly generate input tensors based on the model architecture
    batch_size = 64
    in_channels = 64
    height_in = 128
    width_in = 256
    x = torch.rand(batch_size, in_channels, height_in, width_in).cuda()
    return [x]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    in_channels = 64
    out_channels = 128
    kernel_size = (3, 5)
    return [in_channels, out_channels, kernel_size]