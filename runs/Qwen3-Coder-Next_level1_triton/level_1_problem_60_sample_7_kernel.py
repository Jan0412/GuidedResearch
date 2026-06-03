import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    b_ptr,  # Bias tensor pointer (can be None)
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    in_w, in_h, in_d,  # Input dimensions
    out_w, out_h, out_d,  # Output dimensions
    k_w, k_h, k_d,  # Kernel dimensions
    stride_w, stride_h, stride_d,  # Stride
    pad_w, pad_h, pad_d,  # Padding
    dil_w, dil_h, dil_d,  # Dilation
    groups: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch
    BLOCK_SIZE_K: tl.constexpr,  # Block size for accumulation
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    
    # Calculate output spatial position (simplified for now - will be handled by tiling)
    # We'll process multiple output positions per kernel instance
    # For simplicity, we'll use a basic approach that processes one output position per program
    
    # Compute output position (simplified - we'll handle this with proper tiling later)
    # For now, we'll use a more straightforward implementation
    
    # Calculate the starting positions for this program
    out_idx = pid_batch * (out_w * out_h * out_d) + tl.arange(0, BLOCK_SIZE_N) if BLOCK_SIZE_N > 1 else pid_batch
    
    # Actually, let's implement a more practical approach with proper tiling
    # We'll process one output position per program instance for clarity
    
    # Get output position indices
    out_d_idx = tl.program_id(0) // (out_w * out_h)
    out_h_idx = (tl.program_id(0) % (out_w * out_h)) // out_w
    out_w_idx = (tl.program_id(0) % (out_w * out_h)) % out_w
    
    # For simplicity, let's implement a basic convolution kernel that processes one output at a time
    # This will be more reliable than trying to tile everything at once
    
    if pid_batch >= batch_size:
        return
    
    # Compute input position corresponding to this output position
    in_w_start = out_w_idx * stride_w - pad_w
    in_h_start = out_h_idx * stride_h - pad_h
    in_d_start = out_d_idx * stride_d - pad_d
    
    # Accumulator for the convolution result
    acc = 0.0
    
    # Loop through input channels (grouped if needed)
    for c_in in range(in_channels):
        # Check if this input channel belongs to the current group
        if groups > 1:
            ch_per_group = in_channels // groups
            group_id = c_in // ch_per_group
            out_ch_per_group = out_channels // groups
            if pid_out_ch // out_ch_per_group != group_id:
                continue
        
        # Loop through kernel dimensions
        for k_d_idx in range(k_d):
            in_d_pos = in_d_start + k_d_idx * dil_d
            if in_d_pos < 0 or in_d_pos >= in_d:
                continue
                
            for k_h_idx in range(k_h):
                in_h_pos = in_h_start + k_h_idx * dil_h
                if in_h_pos < 0 or in_h_pos >= in_h:
                    continue
                    
                for k_w_idx in range(k_w):
                    in_w_pos = in_w_start + k_w_idx * dil_w
                    if in_w_pos < 0 or in_w_pos >= in_w:
                        continue
                    
                    # Compute input index
                    input_idx = pid_batch * (in_channels * in_w * in_h * in_d) + \
                               c_in * (in_w * in_h * in_d) + \
                               in_w_pos * (in_h * in_d) + \
                               in_h_pos * in_d + \
                               in_d_pos
                    
                    # Compute weight index
                    weight_idx = pid_out_ch * (in_channels * k_w * k_h * k_d) + \
                                c_in * (k_w * k_h * k_d) + \
                                k_w_idx * (k_h * k_d) + \
                                k_h_idx * k_d + \
                                k_d_idx
                    
                    # Load values and accumulate
                    x_val = tl.load(x_ptr + input_idx)
                    w_val = tl.load(w_ptr + weight_idx)
                    acc += x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        bias_idx = pid_out_ch
        acc += tl.load(b_ptr + bias_idx)
    
    # Store output
    out_idx = pid_batch * (out_channels * out_w * out_h * out_d) + \
             pid_out_ch * (out_w * out_h * out_d) + \
             out_w_idx * (out_h * out_d) + \
             out_h_idx * out_d + \
             out_d_idx
    
    tl.store(out_ptr + out_idx, acc)


def triton_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                  stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 3D convolution.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    batch_size, in_channels, in_w, in_h, in_d = x.shape
    out_channels, _, k_w, k_h, k_d = weight.shape
    
    # Calculate output dimensions
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    
    out_w = (in_w + 2 * padding[0] - dilation[0] * (k_w - 1) - 1) // stride[0] + 1
    out_h = (in_h + 2 * padding[1] - dilation[1] * (k_h - 1) - 1) // stride[1] + 1
    out_d = (in_d + 2 * padding[2] - dilation[2] * (k_d - 1) - 1) // stride[2] + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_w, out_h, out_d, dtype=x.dtype, device=x.device)
    
    # Determine grid dimensions
    # We'll use a 2D grid: (batch_size * out_positions, out_channels)
    out_positions = batch_size * out_w * out_h * out_d
    BLOCK_SIZE_M = 1  # One output channel per block in M dimension
    BLOCK_SIZE_N = 1  # One position per block in N dimension
    
    # Launch kernel
    grid = (out_positions, out_channels)
    conv3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_w, in_h, in_d, out_w, out_h, out_d,
        k_w, k_h, k_d,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create the convolution layer parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights using Kaiming uniform initialization (similar to PyTorch's default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D convolution using Triton kernel.
        """
        return triton_conv3d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, 
            dilation=self.dilation, groups=self.groups
        )


# Import math for sqrt
import math