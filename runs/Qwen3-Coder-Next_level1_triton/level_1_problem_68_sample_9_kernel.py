import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transposed_conv3d_kernel(
    # Pointers to tensors
    input_ptr,      # Input tensor: (B, C_in, D, H, W)
    weight_ptr,     # Weight tensor: (C_in, C_out, Kd, Kh, Kw)
    bias_ptr,       # Bias tensor: (C_out,) or None
    output_ptr,     # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Tensor dimensions
    B, C_in, D, H, W,
    C_out, Kd, Kh, Kw,
    D_out, H_out, W_out,
    # Stride and padding parameters
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    # Block sizes for tiling
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs for output tensor dimensions
    pid_batch = tl.program_id(0)
    pid_cout = tl.program_id(1) // (D_out // BLOCK_SIZE_D)
    pid_d = tl.program_id(1) % (D_out // BLOCK_SIZE_D)
    pid_h = tl.program_id(2) % (H_out // BLOCK_SIZE_H)
    pid_w = tl.program_id(3) % (W_out // BLOCK_SIZE_W)
    
    # Calculate starting positions in output tensor
    out_d_start = pid_d * BLOCK_SIZE_D
    out_h_start = pid_h * BLOCK_SIZE_H
    out_w_start = pid_w * BLOCK_SIZE_W
    
    # Offset for bias if present
    bias_offset = pid_cout * BLOCK_SIZE_COUT
    
    # Allocate accumulator for output
    output_offsets = tl.arange(0, BLOCK_SIZE_D)[:, None, None] * (H_out * W_out) + \
                     tl.arange(0, BLOCK_SIZE_H)[None, :, None] * W_out + \
                     tl.arange(0, BLOCK_SIZE_W)[None, None, :]
    
    # Calculate actual output indices with bounds checking
    out_d_indices = out_d_start + tl.arange(0, BLOCK_SIZE_D)[:, None, None]
    out_h_indices = out_h_start + tl.arange(0, BLOCK_SIZE_H)[None, :, None]
    out_w_indices = out_w_start + tl.arange(0, BLOCK_SIZE_W)[None, None, :]
    
    mask_d = out_d_indices < D_out
    mask_h = out_h_indices < H_out
    mask_w = out_w_indices < W_out
    output_mask = mask_d & mask_h & mask_w
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for start_cin in range(0, C_in, BLOCK_SIZE_CIN):
        # Calculate input channel range
        cin_range = start_cin + tl.arange(0, BLOCK_SIZE_CIN)
        cin_mask = cin_range < C_in
        
        # Calculate corresponding input positions for each output position
        # For transposed convolution: in_d = out_d - (Kd-1) + stride_d * d_offset
        # But we need to check if this maps to a valid input position
        
        # Loop over kernel dimensions
        for kd in range(Kd):
            # Calculate input depth index: in_d = (out_d - out_pad_d) / stride_d - (Kd - 1 - kd) + pad_d
            # Actually for transposed conv: out_d = (in_d - 1) * stride_d - 2*pad_d + out_pad_d + kd
            # So in_d = (out_d - out_pad_d - kd) / stride_d + 1 + pad_d
            
            in_d_base = (out_d_indices - out_pad_d - kd) // stride_d + 1 + pad_d
            in_d_offsets = in_d_base
            
            # Check if in_d is within bounds [0, D)
            in_d_mask = (in_d_offsets >= 0) & (in_d_offsets < D)
            
            for kh in range(Kh):
                in_h_base = (out_h_indices - out_pad_h - kh) // stride_h + 1 + pad_h
                in_h_offsets = in_h_base
                
                in_h_mask = (in_h_offsets >= 0) & (in_h_offsets < H)
                
                for kw in range(Kw):
                    in_w_base = (out_w_indices - out_pad_w - kw) // stride_w + 1 + pad_w
                    in_w_offsets = in_w_base
                    
                    in_w_mask = (in_w_offsets >= 0) & (in_w_offsets < W)
                    
                    # Combined mask for valid input position
                    valid_mask = in_d_mask & in_h_mask & in_w_mask & output_mask
                    
                    # Calculate linear indices for input
                    input_indices = (pid_batch * C_in * D * H * W +
                                    cin_range[:, None, None, None] * D * H * W +
                                    in_d_offsets[None, :, :, :] * H * W +
                                    in_h_offsets[None, :, :, :] * W +
                                    in_w_offsets[None, :, :, :])
                    
                    # Calculate weight indices
                    weight_indices = (cin_range[:, None, None, None] * C_out * Kd * Kh * Kw +
                                    pid_cout * BLOCK_SIZE_COUT * Kd * Kh * Kw +
                                    kd * Kh * Kw * C_out +
                                    tl.arange(0, BLOCK_SIZE_COUT)[None, :, None, None] * Kd * Kh * Kw +
                                    kh * Kw * C_out +
                                    kw * C_out)
                    
                    # Load input values
                    input_offsets = input_indices.reshape(BLOCK_SIZE_CIN, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)
                    input_values = tl.load(input_ptr + input_offsets,
                                          mask=valid_mask[None, :, :, :] & cin_mask[:, None, None, None],
                                          other=0.0)
                    
                    # Load weight values
                    weight_offsets = weight_indices.reshape(BLOCK_SIZE_CIN, BLOCK_SIZE_COUT, 1, 1, 1)
                    weight_values = tl.load(weight_ptr + weight_offsets,
                                          mask=cin_mask[:, None, None, None, None],
                                          other=0.0)
                    
                    # Compute accumulation: output += input * weight
                    # Expand dimensions for broadcasting
                    input_expanded = input_values[:, :, :, :, None]  # (C_in, D, H, W, 1)
                    weight_expanded = weight_values[:, :, None, None, :]  # (C_in, C_out, 1, 1, C_out_block)
                    
                    # Actually, let's simplify: we want to accumulate for each output position
                    # For transposed conv: output[b, cout, od, oh, ow] += sum_cin input[b, cin, id, ih, iw] * weight[cin, cout, kd, kh, kw]
                    
                    # Reshape for proper broadcasting
                    input_flat = input_values.reshape(BLOCK_SIZE_CIN, BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W)
                    weight_flat = weight_values.reshape(BLOCK_SIZE_CIN, BLOCK_SIZE_COUT)
                    
                    # Matrix multiply: (C_in, D*H*W) @ (C_in, C_out) -> (C_in, C_out) but we want (D*H*W, C_out)
                    # Actually: sum over C_in: input[cin, pos] * weight[cin, cout] -> result[pos, cout]
                    temp = tl.dot(input_flat, weight_flat, allow_tf32=True)
                    
                    # Reshape back and accumulate
                    temp_reshaped = temp.reshape(BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_COUT)
                    accumulator += tl.sum(temp_reshaped, axis=3)  # Sum over C_in block
                    
                    # Wait, this approach is getting complex. Let me use a simpler approach:
                    # For each output position, accumulate contributions from all input channels and kernel positions
    
    # Actually, let me rewrite this kernel with a cleaner approach
    
    # Reset for cleaner implementation
    pass  # This is just to keep the structure


@triton.jit
def transposed_conv3d_kernel_v2(
    input_ptr,      # Input tensor: (B, C_in, D, H, W)
    weight_ptr,     # Weight tensor: (C_in, C_out, Kd, Kh, Kw)
    bias_ptr,       # Bias tensor: (C_out,) or None
    output_ptr,     # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Dimensions
    B, C_in, D, H, W,
    C_out, Kd, Kh, Kw,
    D_out, H_out, W_out,
    # Stride and padding
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    # Block sizes
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output position
    out_d = pid_d * BLOCK_SIZE_D
    out_h = pid_h * BLOCK_SIZE_H  
    out_w = pid_w * BLOCK_SIZE_W
    
    # Output indices
    out_d_indices = out_d + tl.arange(0, BLOCK_SIZE_D)[:, None, None]
    out_h_indices = out_h + tl.arange(0, BLOCK_SIZE_H)[None, :, None]
    out_w_indices = out_w + tl.arange(0, BLOCK_SIZE_W)[None, None, :]
    
    # Masks for output bounds
    mask_d = out_d_indices < D_out
    mask_h = out_h_indices < H_out
    mask_w = out_w_indices < W_out
    output_mask = mask_d & mask_h & mask_w
    
    # Accumulator
    output_vals = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over kernel positions
    for kd in range(Kd):
        # Calculate corresponding input depth: in_d = (out_d - out_pad_d - kd) / stride_d + pad_d
        # Only valid if (out_d - out_pad_d - kd) is divisible by stride_d and in_d is in bounds
        in_d_base = (out_d_indices - out_pad_d - kd) 
        in_d_valid = (in_d_base >= 0) & ((in_d_base % stride_d) == 0)
        in_d = in_d_base // stride_d + pad_d
        
        for kh in range(Kh):
            in_h_base = (out_h_indices - out_pad_h - kh)
            in_h_valid = (in_h_base >= 0) & ((in_h_base % stride_h) == 0)
            in_h = in_h_base // stride_h + pad_h
            
            for kw in range(Kw):
                in_w_base = (out_w_indices - out_pad_w - kw)
                in_w_valid = (in_w_base >= 0) & ((in_w_base % stride_w) == 0)
                in_w = in_w_base // stride_w + pad_w
                
                # Combined validity mask
                valid = in_d_valid & in_h_valid & in_w_valid & output_mask & (in_d < D) & (in_h < H) & (in_w < W)
                
                # Calculate input linear indices
                input_offsets = (pid_batch * C_in * D * H * W +
                               tl.arange(0, C_in)[:, None, None, None] * D * H * W +
                               in_d[None, :, :, :] * H * W +
                               in_h[None, :, :, :] * W +
                               in_w[None, :, :, :])
                
                # Calculate weight linear indices
                weight_offsets = (tl.arange(0, C_in)[:, None, None, None] * C_out * Kd * Kh * Kw +
                                 pid_cout * BLOCK_SIZE_COUT * Kd * Kh * Kw +
                                 kd * Kh * Kw * C_out +
                                 tl.arange(0, BLOCK_SIZE_COUT)[None, :, None, None] * Kd * Kh * Kw +
                                 kh * Kw * C_out +
                                 kw * C_out)
                
                # Load input and weight
                input_values = tl.load(input_ptr + input_offsets,
                                      mask=valid[None, :, :, :] & (tl.arange(0, C_in) < C_in)[:, None, None, None],
                                      other=0.0)
                weight_values = tl.load(weight_ptr + weight_offsets,
                                       mask=(tl.arange(0, C_in) < C_in)[:, None, None, None] & 
                                             (tl.arange(0, BLOCK_SIZE_COUT) < C_out)[None, :, None, None],
                                       other=0.0)
                
                # Accumulate: sum over C_in of input[cin] * weight[cin, cout]
                # input: (C_in, D, H, W), weight: (C_in, C_out_block, 1, 1)
                # result: (D, H, W, C_out_block)
                
                # Reshape for dot product
                input_flat = input_values.reshape(C_in, BLOCK_SIZE_D * BLOCK_SIZE_H * BLOCK_SIZE_W)
                weight_flat = weight_values.reshape(C_in, BLOCK_SIZE_COUT)
                
                # Matrix multiply: (C_in, D*H*W) @ (C_in, C_out) -> need to transpose weight
                # Actually: sum over C_in: input[cin, pos] * weight[cin, cout] -> result[pos, cout]
                temp = tl.dot(input_flat, weight_flat, allow_tf32=True)
                
                # Reshape to 4D and accumulate
                temp_reshaped = temp.reshape(BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_COUT)
                output_vals += tl.sum(temp_reshaped, axis=3)  # Sum over C_in block
                
                # Wait, this is still not quite right. Let me fix the matrix multiplication.
    
    # Actually, let me implement a much cleaner version that processes one output position at a time
    pass


@triton.jit
def transposed_conv3d_fused(
    input_ptr,      # Input tensor: (B, C_in, D, H, W)
    weight_ptr,     # Weight tensor: (C_in, C_out, Kd, Kh, Kw)
    bias_ptr,       # Bias tensor: (C_out,) or None
    output_ptr,     # Output tensor: (B, C_out, D_out, H_out, W_out)
    # Dimensions
    B, C_in, D, H, W,
    C_out, Kd, Kh, Kw,
    D_out, H_out, W_out,
    # Stride and padding
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    # Block sizes
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Calculate output position
    out_d = pid_d * BLOCK_SIZE_D
    out_h = pid_h * BLOCK_SIZE_H  
    out_w = pid_w * BLOCK_SIZE_W
    
    # Output indices
    out_d_indices = out_d + tl.arange(0, BLOCK_SIZE_D)[:, None, None]
    out_h_indices = out_h + tl.arange(0, BLOCK_SIZE_H)[None, :, None]
    out_w_indices = out_w + tl.arange(0, BLOCK_SIZE_W)[None, None, :]
    
    # Masks for output bounds
    mask_d = out_d_indices < D_out
    mask_h = out_h_indices < H_out
    mask_w = out_w_indices < W_out
    output_mask = mask_d & mask_h & mask_w
    
    # Accumulator
    output_vals = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input channels
    for cin in range(C_in):
        # Loop over kernel positions
        for kd in range(Kd):
            # Calculate corresponding input depth
            in_d = (out_d_indices - out_pad_d - kd) // stride_d + pad_d
            in_d_valid = ((out_d_indices - out_pad_d - kd) % stride_d == 0) & (in_d >= 0) & (in_d < D)
            
            for kh in range(Kh):
                in_h = (out_h_indices - out_pad_h - kh) // stride_h + pad_h
                in_h_valid = ((out_h_indices - out_pad_h - kh) % stride_h == 0) & (in_h >= 0) & (in_h < H)
                
                for kw in range(Kw):
                    in_w = (out_w_indices - out_pad_w - kw) // stride_w + pad_w
                    in_w_valid = ((out_w_indices - out_pad_w - kw) % stride_w == 0) & (in_w >= 0) & (in_w < W)
                    
                    # Combined validity mask
                    valid = in_d_valid & in_h_valid & in_w_valid & output_mask
                    
                    # Calculate input linear offset
                    input_offset = (pid_batch * C_in * D * H * W +
                                  cin * D * H * W +
                                  in_d * H * W +
                                  in_h * W +
                                  in_w)
                    
                    # Calculate weight linear offset
                    weight_offset = (cin * C_out * Kd * Kh * Kw +
                                   pid_cout * Kd * Kh * Kw +
                                   kd * Kh * Kw +
                                   kh * Kw +
                                   kw)
                    
                    # Load input value
                    input_val = tl.load(input_ptr + input_offset, mask=valid, other=0.0)
                    # Load weight value
                    weight_val = tl.load(weight_ptr + weight_offset, mask=valid, other=0.0)
                    
                    # Accumulate
                    output_vals += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_offset = pid_cout
        bias_val = tl.load(bias_ptr + bias_offset)
        output_vals += bias_val
    
    # Store output
    output_offset = (pid_batch * C_out * D_out * H_out * W_out +
                    pid_cout * D_out * H_out * W_out +
                    out_d_indices * H_out * W_out +
                    out_h_indices * W_out +
                    out_w_indices)
    tl.store(output_ptr + output_offset, output_vals.to(input_ptr.dtype.element_ty), mask=output_mask)


def triton_transposed_conv3d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1, 1),
    padding: tuple = (0, 0, 0),
    output_padding: tuple = (0, 0, 0),
    groups: int = 1,
):
    """
    Performs transposed 3D convolution using Triton kernel.
    
    Args:
        input_tensor: Input tensor of shape (B, C_in, D, H, W)
        weight: Weight tensor of shape (C_in, C_out, Kd, Kh, Kw)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride tuple (stride_d, stride_h, stride_w)
        padding: Padding tuple (pad_d, pad_h, pad_w)
        output_padding: Output padding tuple (out_pad_d, out_pad_h, out_pad_w)
        groups: Number of groups (must be 1 for this implementation)
    
    Returns:
        Output tensor of shape (B, C_out, D_out, H_out, W_out)
    """
    assert input_tensor.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert groups == 1, "Only groups=1 is supported in this Triton kernel."
    
    # Ensure contiguous tensors
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Extract dimensions
    B, C_in, D, H, W = input_tensor.shape
    C_in_w, C_out, Kd, Kh, Kw = weight.shape
    
    assert C_in == C_in_w, f"Input channels {C_in} doesn't match weight input channels {C_in_w}"
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding
    
    D_out = (D - 1) * stride_d - 2 * pad_d + out_pad_d + Kd
    H_out = (H - 1) * stride_h - 2 * pad_h + out_pad_h + Kh
    W_out = (W - 1) * stride_w - 2 * pad_w + out_pad_w + Kw
    
    # Create output tensor
    output = torch.empty(B, C_out, D_out, H_out, W_out, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE_COUT = 8  # Block size for output channels
    BLOCK_SIZE_D = 4     # Block size for depth
    BLOCK_SIZE_H = 4     # Block size for height
    BLOCK_SIZE_W = 4     # Block size for width
    
    grid = lambda meta: (
        B,  # batch dimension
        triton.cdiv(C_out, BLOCK_SIZE_COUT),  # output channels
        triton.cdiv(D_out, BLOCK_SIZE_D),     # depth
        triton.cdiv(H_out, BLOCK_SIZE_H),     # height
        triton.cdiv(W_out, BLOCK_SIZE_W),     # width
    )
    
    # Launch kernel
    transposed_conv3d_fused[grid](
        input_tensor, weight, bias, output,
        B, C_in, D, H, W,
        C_out, Kd, Kh, Kw,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for transposed 3D convolution.
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
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using our Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return triton_transposed_conv3d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, 
            output_padding=self.output_padding, groups=self.groups
        )


# Import math for initialization
import math