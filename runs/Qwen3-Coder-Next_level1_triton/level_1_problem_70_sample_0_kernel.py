import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Pointers to inputs and outputs
    x_ptr,  # Input tensor: [B, C_in, D, H, W]
    w_ptr,  # Weight tensor: [C_in, C_out, Kd, Kh, Kw]
    b_ptr,  # Bias tensor: [C_out] (can be None)
    out_ptr,  # Output tensor: [B, C_out, D_out, H_out, W_out]
    # Sizes
    B, C_in, C_out,
    D, H, W,
    D_out, H_out, W_out,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    dil_d, dil_h, dil_w,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # Block size for C_out dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for input channel dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for kernel elements
):
    # Get program IDs
    pid_b = tl.program_id(0)  # Batch index
    pid_c = tl.program_id(1)  # Output channel block index
    
    # Calculate output channel range for this program
    c_offsets = pid_c * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_mask = c_offsets < C_out
    
    # Initialize accumulator for output values
    output = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Iterate over input channels in blocks
    for c_in_start in range(0, C_in, BLOCK_SIZE_N):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_N)
        c_in_mask = c_in_offsets < C_in
        
        # Iterate over kernel elements in blocks
        for k_d in range(0, Kd, BLOCK_SIZE_K):
            for k_h in range(0, Kh, BLOCK_SIZE_K):
                for k_w in range(0, Kw, BLOCK_SIZE_K):
                    # Calculate kernel indices
                    k_d_offsets = k_d + tl.arange(0, BLOCK_SIZE_K)
                    k_h_offsets = k_h + tl.arange(0, BLOCK_SIZE_K)
                    k_w_offsets = k_w + tl.arange(0, BLOCK_SIZE_K)
                    
                    # Compute input indices from output indices
                    # We'll process one output location at a time (simplified approach)
                    pass  # We need to restructure this for efficiency
                    
    # Since the direct approach is complex, we use a more efficient blocked approach
    # Let's compute output for one output location per program and use shared memory


# Alternative: Use a more practical kernel that processes output locations
@triton.jit
def conv_transpose3d_simple_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, C_out,
    D, H, W,
    D_out, H_out, W_out,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    dil_d, dil_h, dil_w,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs: 
    # pid_b = batch index
    # pid_d = output depth index
    # pid_h = output height index  
    # pid_w = output width index
    # pid_c = output channel block index
    
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    pid_c = tl.program_id(4)
    
    # Calculate output channel range
    c_out_start = pid_c * BLOCK_SIZE_C_OUT
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out_offsets < C_out
    
    # Compute input position for this output position (transposed conv mapping)
    # For transposed convolution: input_pos = output_pos * stride - pad + kernel_pos * dilation
    
    # Initialize output accumulator
    output = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=tl.float32)
    
    # Iterate over all input channels
    for c_in in range(C_in):
        # Iterate over kernel elements
        for k_d in range(Kd):
            for k_h in range(Kh):
                for k_w in range(Kw):
                    # Calculate corresponding input position
                    in_d = pid_d * stride_d - pad_d + k_d * dil_d
                    in_h = pid_h * stride_h - pad_h + k_h * dil_h
                    in_w = pid_w * stride_w - pad_w + k_w * dil_w
                    
                    # Check if input position is valid
                    valid_input = (in_d >= 0) & (in_d < D) & \
                                  (in_h >= 0) & (in_h < H) & \
                                  (in_w >= 0) & (in_w < W)
                    
                    if tl.static_bool(valid_input):
                        # Calculate input index
                        input_idx = ((pid_b * C_in + c_in) * D + in_d) * H * W + \
                                   in_h * W + in_w
                        # Calculate weight index: w[c_in, c_out, k_d, k_h, k_w]
                        weight_idx = ((c_in * C_out + c_out_offsets) * Kd + k_d) * Kh * Kw + \
                                    k_h * Kw + k_w
                        
                        # Load input and weight values
                        x_val = tl.load(x_ptr + input_idx, mask=valid_input, other=0.0)
                        w_vals = tl.load(w_ptr + weight_idx, mask=c_out_mask, other=0.0)
                        
                        # Accumulate: output[c_out] += x[input] * weight[c_in, c_out, k_d, k_h, k_w]
                        output += x_val * w_vals.to(tl.float32)
    
    # Apply bias if present
    if b_ptr is not None:
        bias_idx = c_out_offsets
        bias_vals = tl.load(b_ptr + bias_idx, mask=c_out_mask, other=0.0)
        output += bias_vals.to(tl.float32)
    
    # Store result
    output_idx = ((pid_b * C_out + c_out_offsets) * D_out + pid_d) * H_out * W_out + \
                pid_h * W_out + pid_w
    tl.store(out_ptr + output_idx, output, mask=c_out_mask)


# Even better: use a blocked approach with shared memory for better performance
@triton.jit
def conv_transpose3d_optimized_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, C_out,
    D, H, W,
    D_out, H_out, W_out,
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    dil_d, dil_h, dil_w,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs for output locations and channel blocks
    pid_b = tl.program_id(0)
    pid_cd = tl.program_id(1)  # Combined index for D_out * H_out * W_out
    pid_c = tl.program_id(2)
    
    # Decode pid_cd into d, h, w indices
    hw_out = H_out * W_out
    pid_d = pid_cd // hw_out
    rem = pid_cd % hw_out
    pid_h = rem // W_out
    pid_w = rem % W_out
    
    # Output channel block
    c_out_start = pid_c * BLOCK_SIZE_C_OUT
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_OUT)
    c_out_mask = c_out_offsets < C_out
    
    # Shared memory buffers for input and weights
    # For simplicity, we'll use a direct computation without shared memory
    # since the kernel is relatively small (typically 3x3x3)
    
    output = tl.zeros((BLOCK_SIZE_C_OUT,), dtype=tl.float32)
    
    # Iterate over input channels and kernel elements
    for c_in in range(C_in):
        for k_d in range(Kd):
            for k_h in range(Kh):
                for k_w in range(Kw):
                    # Calculate input position
                    in_d = pid_d * stride_d - pad_d + k_d * dil_d
                    in_h = pid_h * stride_h - pad_h + k_h * dil_h
                    in_w = pid_w * stride_w - pad_w + k_w * dil_w
                    
                    # Check bounds
                    valid = (in_d >= 0) & (in_d < D) & \
                           (in_h >= 0) & (in_h < H) & \
                           (in_w >= 0) & (in_w < W)
                    
                    if tl.static_bool(valid):
                        # Calculate memory indices
                        x_idx = ((pid_b * C_in + c_in) * D + in_d) * H * W + in_h * W + in_w
                        w_idx = ((c_in * C_out + c_out_offsets) * Kd + k_d) * Kh * Kw + k_h * Kw + k_w
                        
                        # Load and multiply-accumulate
                        x_val = tl.load(x_ptr + x_idx)
                        w_vals = tl.load(w_ptr + w_idx, mask=c_out_mask)
                        output += x_val * w_vals.to(tl.float32)
    
    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_offsets, mask=c_out_mask)
        output += bias.to(tl.float32)
    
    # Store result
    out_idx = ((pid_b * C_out + c_out_offsets) * D_out + pid_d) * H_out * W_out + pid_h * W_out + pid_w
    tl.store(out_ptr + out_idx, output, mask=c_out_mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Triton implementation of 3D transposed convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert x.dim() == 5, "Input must be 5D tensor (B, C, D, H, W)"
    assert weight.dim() == 5, "Weight must be 5D tensor (C_in, C_out, Kd, Kh, Kw)"
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Extract dimensions
    B, C_in, D, H, W = x.shape
    C_in_w, C_out, Kd, Kh, Kw = weight.shape
    assert C_in == C_in_w, f"Input channels {C_in} doesn't match weight input channels {C_in_w}"
    
    # Handle stride, padding, output_padding, dilation as tuples if needed
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_d = pad_h = pad_w = padding
    else:
        pad_d, pad_h, pad_w = padding
        
    if isinstance(output_padding, int):
        out_pad_d = out_pad_h = out_pad_w = output_padding
    else:
        out_pad_d, out_pad_h, out_pad_w = output_padding
        
    if isinstance(dilation, int):
        dil_d = dil_h = dil_w = dilation
    else:
        dil_d, dil_h, dil_w = dilation
    
    # Calculate output dimensions
    D_out = (D - 1) * stride_d - 2 * pad_d + dil_d * (Kd - 1) + out_pad_d + 1
    H_out = (H - 1) * stride_h - 2 * pad_h + dil_h * (Kh - 1) + out_pad_h + 1
    W_out = (W - 1) * stride_w - 2 * pad_w + dil_w * (Kw - 1) + out_pad_w + 1
    
    # Create output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Set block sizes for kernel
    BLOCK_SIZE_C_OUT = 16
    BLOCK_SIZE_C_IN = 16
    BLOCK_SIZE_K = 3  # Since kernel is typically small (3x3x3)
    
    # Calculate grid dimensions
    # We use one block per output location (D_out * H_out * W_out) * batch
    grid = (B, D_out * H_out * W_out, (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT)
    
    # Launch kernel
    conv_transpose3d_optimized_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out,
        D, H, W,
        D_out, H_out, W_out,
        Kd, Kh, Kw,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        dil_d, dil_h, dil_w,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 3D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters but without the actual layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weights and bias (same as nn.ConvTranspose3d)
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Kaiming uniform initialization like PyTorch's ConvTranspose3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using our custom Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )


# Import math for the initialization
import math