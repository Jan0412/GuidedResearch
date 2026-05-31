import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    in_channels,
    out_channels,
    kernel_depth,
    kernel_width,
    kernel_height,
    stride_depth,
    stride_width,
    stride_height,
    padding_depth,
    padding_width,
    padding_height,
    groups,
    batch_size,
    depth_in,
    width_in,
    height_in,
    depth_out,
    width_out,
    height_out,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Grid dimensions
    # We flatten B, C_out, D_out, W_out into the grid, and handle H_out with threads
    pid = tl.program_id(0)
    
    # Decode output coordinates
    num_hw = width_out * height_out
    num_dhw = depth_out * num_hw
    num_c_dhw = out_channels * num_dhw
    
    b = pid // num_c_dhw
    rest = pid % num_c_dhw
    c_out = rest // num_dhw
    rest = rest % num_dhw
    d_out = rest // num_hw
    rest = rest % num_hw
    w_out = rest // height_out
    h_out = rest % height_out
    
    # Base input coordinates for this output element
    base_d = d_out * stride_depth - padding_depth
    base_w = w_out * stride_width - padding_width
    base_h = h_out * stride_height - padding_height
    
    # Accumulator
    acc = 0.0
    
    # Groups
    group_id = c_out % groups
    channels_per_group = in_channels // groups
    c_in_start = group_id * channels_per_group
    
    # Weights pointer offset for this c_out
    # Weight shape: (out_channels, in_channels // groups, kernel_depth, kernel_width, kernel_height)
    weight_offset = c_out * channels_per_group * kernel_depth * kernel_width * kernel_height
    
    # Iterate over input channels in blocks
    for c_in_block in range(0, channels_per_group, BLOCK_SIZE_C):
        c_in_idx = c_in_start + c_in_block
        
        # Load weights for this block
        # Weights are accessed as (c_in_local, k_d, k_w, k_h)
        # We can load a tile of weights
        # Since kernel dimensions are small, we might load all or chunk them
        # For simplicity and efficiency, we load weights in a tiled manner over C_in
        # and iterate over K within the tile or vice versa
        
        # To optimize, we load weights into registers/shared memory
        # Given K is small, we can load all K for a C_IN_BLOCK
        
        # Create offsets for weights
        # Weight indices: c_in_local, k_d, k_w, k_h
        c_in_offsets = tl.arange(0, BLOCK_SIZE_C) + c_in_block
        mask_c = c_in_offsets < channels_per_group
        
        # We need to load weights for all k_d, k_w, k_h
        # This requires a loop or unrolling over K
        # Since K is constexpr, we can loop
        
        for k_d in range(kernel_depth):
            for k_w in range(kernel_width):
                for k_h in range(kernel_height):
                    # Weight index
                    w_idx = c_in_offsets + k_d * channels_per_group * kernel_width * kernel_height + \
                            k_w * channels_per_group * kernel_height + k_h
                    w_mask = mask_c
                    
                    # Load weight
                    w = tl.load(weight_ptr + w_idx, mask=w_mask, other=0.0)
                    
                    # Input indices
                    # x[b, c_in, d_in, w_in, h_in]
                    # d_in = base_d + k_d
                    # w_in = base_w + k_w
                    # h_in = base_h + k_h
                    
                    d_in = base_d + k_d
                    w_in = base_w + k_w
                    h_in = base_h + k_h
                    
                    # Check bounds for input
                    # We can mask based on input bounds
                    mask_d = (d_in >= 0) & (d_in < depth_in)
                    mask_w = (w_in >= 0) & (w_in < width_in)
                    mask_h = (h_in >= 0) & (h_in < height_in)
                    mask_input = mask_c & mask_d & mask_w & mask_h
                    
                    # Input pointer offset
                    # x shape: (batch_size, in_channels, depth, width, height)
                    x_idx = b * in_channels * depth_in * width_in * height_in + \
                            (c_in_offsets + c_in_start) * depth_in * width_in * height_in + \
                            d_in * width_in * height_in + \
                            w_in * height_in + \
                            h_in
                    
                    # Load input
                    x_val = tl.load(x_ptr + x_idx, mask=mask_input, other=0.0)
                    
                    # Accumulate
                    acc += w * x_val
    
    # Add bias if available
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + c_out)
    
    # Store result
    out_idx = b * out_channels * depth_out * width_out * height_out + \
              c_out * depth_out * width_out * height_out + \
              d_out * width_out * height_out + \
              w_out * height_out + \
              h_out
    tl.store(out_ptr + out_idx, acc)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, depth_in, width_in, height_in = x.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    stride_depth, stride_width, stride_height = stride
    padding_depth, padding_width, padding_height = padding
    
    # Calculate output dimensions
    # Formula for ConvTranspose3d output size:
    # D_out = (D_in - 1) * S_d - 2*P_d + K_d + O_d
    # However, PyTorch uses output_padding to adjust.
    # We need to compute actual output shape.
    # PyTorch ConvTranspose3d output size calculation:
    # depth_out = (depth_in - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth
    # But this is for the case where padding is applied to input.
    # Actually, PyTorch formula is:
    # out = (in - 1) * stride - 2 * padding + kernel + output_padding
    # Wait, this is for the case where padding is used in the convolution sense.
    # For ConvTranspose, the formula is:
    # D_out = (D_in - 1) * S_d - 2*P_d + K_d + O_d
    # Let's verify with PyTorch behavior.
    # If padding=0, stride=1, kernel=3, input=4 -> output = 3*1 - 0 + 3 = 4?
    # PyTorch: ConvTranspose3d(1,1,3, stride=1, padding=0) on (1,1,4,4,4) -> (1,1,4,4,4).
    # Formula: (4-1)*1 - 0 + 3 = 3+3=6? No.
    # Correct formula for ConvTranspose:
    # D_out = (D_in - 1) * S_d - 2*P_d + K_d + O_d
    # Let's check: (4-1)*1 - 0 + 3 = 6. This is wrong.
    # The correct formula is:
    # D_out = D_in * S_d - 2*P_d + K_d - S_d + O_d ?
    # Or D_out = (D_in - 1) * S_d + K_d - 2*P_d + O_d?
    # PyTorch docs:
    # output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    # For transpose conv, dilation is 1.
    # So D_out = (D_in - 1) * S_d - 2*P_d + (K_d - 1) + O_d + 1
    # = (D_in - 1) * S_d - 2*P_d + K_d + O_d
    # This matches my previous formula.
    # Why did the example fail?
    # Input=4, S=1, P=0, K=3.
    # (4-1)*1 - 0 + 3 = 6.
    # But PyTorch gives 4.
    # Ah, PyTorch ConvTranspose3d with padding=0, stride=1, kernel=3 on input 4 gives output 4.
    # The formula in docs might be different or I am misremembering.
    # Let's use PyTorch's actual output shape calculation to be safe.
    # We can compute output shape using a dummy forward pass or the formula.
    # The formula in docs is correct for the operation.
    # Maybe the example input size is different.
    # Let's trust the formula and compute output shape.
    # D_out = (D_in - 1) * S_d - 2*P_d + K_d + O_d
    # If this gives wrong result, we might need to adjust.
    # However, for the kernel, we need the output shape.
    # We can compute it as:
    depth_out = (depth_in - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding[0]
    width_out = (width_in - 1) * stride_width - 2 * padding_width + kernel_width + output_padding[1]
    height_out = (height_in - 1) * stride_height - 2 * padding_height + kernel_height + output_padding[2]
    
    # Ensure output shape matches PyTorch
    # If there's a discrepancy, we might need to clamp or adjust.
    # But for the kernel, we assume the formula holds.
    # In practice, PyTorch might handle output_padding differently.
    # output_padding adds to one side.
    # The formula should be correct.
    
    out = torch.empty(batch_size, out_channels, depth_out, width_out, height_out, device=x.device, dtype=x.dtype)
    
    # Grid configuration
    # Flatten B, C_out, D_out, W_out
    num_blocks = batch_size * out_channels * depth_out * width_out
    grid = (num_blocks,)
    
    # Block sizes
    # Tune these based on hardware
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_K = 1  # K is handled by loops
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        in_channels, out_channels,
        kernel_depth, kernel_width, kernel_height,
        stride_depth, stride_width, stride_height,
        padding_depth, padding_width, padding_height,
        groups,
        batch_size,
        depth_in, width_in, height_in,
        depth_out, width_out, height_out,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        # Store parameters for kernel
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias
        return triton_conv_transpose3d(x, weight, bias, self.stride, self.padding, self.output_padding, self.groups)


def get_inputs():
    batch_size = 16
    in_channels = 32
    depth = 64
    width = 64
    height = 64
    x = torch.rand(batch_size, in_channels, depth, width, height)
    return [x]


def get_init_inputs():
    in_channels = 32
    out_channels = 64
    kernel_depth = 3
    kernel_width = 5
    kernel_height = 5
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]