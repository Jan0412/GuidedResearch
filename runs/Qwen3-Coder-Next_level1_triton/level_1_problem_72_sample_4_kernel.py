import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transposed_conv3d_kernel(
    # Pointers to inputs and outputs
    X_ptr,  # Input tensor: (B, C_in, D_in, H_in, W_in)
    W_ptr,  # Weight tensor: (C_in, C_out // groups, K_d, K_h, K_w)
    B_ptr,  # Bias tensor: (C_out,) or None
    Y_ptr,  # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Dimensions
    batch_size, in_channels, out_channels,
    depth_in, height_in, width_in,
    depth_out, height_out, width_out,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    groups,
    # Block sizes for parallelization
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch size
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
    BLOCK_D: tl.constexpr,       # Block size for depth
    BLOCK_H: tl.constexpr,       # Block size for height
    BLOCK_W: tl.constexpr,       # Block size for width
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output position
    out_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create masks for valid indices
    mask_d = out_d < depth_out
    mask_h = out_h < height_out
    mask_w = out_w < width_out
    mask_hw = mask_h[:, None] & mask_w[None, :]
    mask_dhw = mask_d[:, None, None] & mask_hw[None, :, :]
    
    # Calculate starting positions for input
    in_d = (out_d * stride_d - pad_d)[:, None, None]
    in_h = (out_h * stride_h - pad_h)[None, :, None]
    in_w = (out_w * stride_w - pad_w)[None, None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for k_d in range(kernel_d):
        for k_h in range(kernel_h):
            for k_w in range(kernel_w):
                # Calculate input position for this kernel element
                in_d_pos = in_d + k_d
                in_h_pos = in_h + k_h
                in_w_pos = in_w + k_w
                
                # Mask for valid input positions
                valid_mask = (in_d_pos >= 0) & (in_d_pos < depth_in) & \
                            (in_h_pos >= 0) & (in_h_pos < height_in) & \
                            (in_w_pos >= 0) & (in_w_pos < width_in)
                
                # Calculate indices for input tensor
                # Input shape: (batch_size, in_channels, depth_in, height_in, width_in)
                input_indices = pid_batch * (in_channels * depth_in * height_in * width_in) + \
                               tl.arange(0, BLOCK_SIZE_K)[:, None, None, None] * (depth_in * height_in * width_in) + \
                               in_d_pos[None, :, :, :] * (height_in * width_in) + \
                               in_h_pos[None, :, :, :] * width_in + \
                               in_w_pos[None, :, :, :]
                
                # Calculate indices for weight tensor
                # Weight shape: (in_channels, out_channels // groups, kernel_d, kernel_h, kernel_w)
                weight_indices = tl.arange(0, BLOCK_SIZE_K)[:, None, None, None] * (out_channels // groups * kernel_d * kernel_h * kernel_w) + \
                                pid_c_out * (kernel_d * kernel_h * kernel_w) + \
                                k_d * (kernel_h * kernel_w) + \
                                k_h * kernel_w + \
                                k_w
                
                # Load data (only valid positions)
                input_vals = tl.load(X_ptr + input_indices, 
                                    mask=valid_mask[None, :, :, :] & 
                                        (tl.arange(0, BLOCK_SIZE_K)[:, None, None, None] < in_channels),
                                    other=0.0)
                weight_vals = tl.load(W_ptr + weight_indices, 
                                    mask=tl.arange(0, BLOCK_SIZE_K)[:, None, None, None] < in_channels,
                                    other=0.0)
                
                # Accumulate convolution result
                acc += tl.sum(input_vals * weight_vals, axis=0)
    
    # Add bias if present
    if B_ptr is not None:
        bias = tl.load(B_ptr + pid_c_out * (out_channels // groups) + tl.arange(0, BLOCK_SIZE_M) % (out_channels // groups))
        acc += bias[None, :, :, :]
    
    # Store output
    # Output shape: (batch_size, out_channels, depth_out, height_out, width_out)
    out_indices = pid_batch * (out_channels * depth_out * height_out * width_out) + \
                 pid_c_out * (depth_out * height_out * width_out) + \
                 out_d[:, None, None] * (height_out * width_out) + \
                 out_h[None, :, None] * width_out + \
                 out_w[None, None, :]
    
    # Apply mask and store
    tl.store(Y_ptr + out_indices, acc, mask=mask_dhw)


def triton_transposed_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    output_padding=(0, 0, 0),
    groups=1,
):
    """
    Custom Triton implementation of 3D transposed convolution.
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Get dimensions
    batch_size, in_channels, depth_in, height_in, width_in = x.shape
    _, out_channels, kernel_d, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    depth_out = (depth_in - 1) * stride[0] - 2 * padding[0] + kernel_d + output_padding[0]
    height_out = (height_in - 1) * stride[1] - 2 * padding[1] + kernel_h + output_padding[1]
    width_out = (width_in - 1) * stride[2] - 2 * padding[2] + kernel_w + output_padding[2]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, depth_out, height_out, width_out, 
                        dtype=x.dtype, device=x.device)
    
    # Set up grid dimensions for kernel
    # Grid: (batch_size, out_channels // BLOCK_SIZE_M, depth_out // BLOCK_D, height_out // BLOCK_H, width_out // BLOCK_W)
    # Using reasonable block sizes for FP32
    BLOCK_SIZE_M = 8   # Output channels per block
    BLOCK_SIZE_N = 1   # Batch per block
    BLOCK_SIZE_K = 16  # Input channels per block
    BLOCK_D = 4        # Depth per block
    BLOCK_H = 4        # Height per block
    BLOCK_W = 4        # Width per block
    
    grid = (
        batch_size,
        (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (depth_out + BLOCK_D - 1) // BLOCK_D,
        (height_out + BLOCK_H - 1) // BLOCK_H,
        (width_out + BLOCK_W - 1) // BLOCK_W,
    )
    
    # Launch kernel
    transposed_conv3d_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, out_channels,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D transposed convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weight and bias (same as original nn.ConvTranspose3d)
        # Weight shape: (in_channels, out_channels // groups, kernel_d, kernel_h, kernel_w)
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, 
                       kernel_size[0], kernel_size[1], kernel_size[2])
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming initialization similar to PyTorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D transposed convolution using Triton kernel.
        """
        return triton_transposed_conv3d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding, self.groups
        )