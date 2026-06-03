import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer (N, C_in, H, W)
    w_ptr,  # Weight tensor pointer (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,) - can be None
    out_ptr,  # Output tensor pointer (N, C_out, H_out, W_out)
    N, C_in, H, W,  # Input dimensions
    C_out, K_h, K_w,  # Output and kernel dimensions
    stride_h, stride_w,  # Stride
    padding_h, padding_w,  # Padding
    dilation_h, dilation_w,  # Dilation
    C_in_Kh_Kw,  # C_in * K_h * K_w
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)  # Batch index
    pid_c_out = tl.program_id(1)  # Output channel index
    
    # Calculate output spatial dimensions
    H_out = (H + 2 * padding_h - dilation_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * padding_w - dilation_w * (K_w - 1) - 1) // stride_w + 1
    
    # Calculate output position
    # We'll compute one output element per program (or a small tile)
    # For simplicity, each program computes one output element
    # In a more optimized version, we'd use tiles
    
    # Calculate output position
    h_out = pid_c_out // C_out
    c_out = pid_c_out % C_out
    
    # But we need to handle the case where pid_c_out > H_out * C_out
    # Let's restructure: we'll use a 3D grid where:
    # pid_n = batch, pid_h = output_h, pid_c_out = output channel
    
    # Actually, let's restructure the kernel for better efficiency
    # Using a 3D grid: (batch, output_h, output_channel)
    
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c_out = tl.program_id(2)
    
    if pid_batch >= N or pid_h >= H_out or pid_c_out >= C_out:
        return
    
    # Calculate output position
    out_h = pid_h
    out_w_start = 0  # We'll iterate over output_w in a loop
    out_w_end = W_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_W,), tl.float32) if tl.constexpr hasattr(tl, 'BLOCK_SIZE_W') else tl.zeros((1,), tl.float32)
    
    # For simplicity, compute one output element per program
    # In a production kernel, we'd use tiling
    
    # Compute the convolution for this output element
    output_val = 0.0
    
    # Iterate over input channels
    for c_in in range(C_in):
        # Iterate over kernel height
        for kh in range(K_h):
            # Calculate input h position
            in_h = out_h * stride_h - padding_h + kh * dilation_h
            
            # Check if input h is valid
            if in_h < 0 or in_h >= H:
                continue
                
            # Iterate over kernel width
            for kw in range(K_w):
                # Calculate input w position
                in_w = out_w_start * stride_w - padding_w + kw * dilation_w
                
                # Check if input w is valid
                if in_w < 0 or in_w >= W:
                    continue
                    
                # Calculate input pointer offset
                x_offset = (pid_batch * C_in * H * W + 
                           c_in * H * W + 
                           in_h * W + 
                           in_w)
                
                # Calculate weight pointer offset
                w_offset = (pid_c_out * C_in * K_h * K_w + 
                           c_in * K_h * K_w + 
                           kh * K_w + 
                           kw)
                
                x_val = tl.load(x_ptr + x_offset)
                w_val = tl.load(w_ptr + w_offset)
                
                output_val += x_val * w_val
    
    # Add bias if available
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + pid_c_out)
        output_val += bias_val
    
    # Calculate output position
    out_offset = (pid_batch * C_out * H_out * W_out + 
                 pid_c_out * H_out * W_out + 
                 out_h * W_out + 
                 out_w_start)
    
    tl.store(out_ptr + out_offset, output_val)


# Better implementation with tiling for efficiency
@triton.jit
def conv2d_kernel_optimized(
    x_ptr,  # Input tensor pointer (N, C_in, H, W)
    w_ptr,  # Weight tensor pointer (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,) - can be None
    out_ptr,  # Output tensor pointer (N, C_out, H_out, W_out)
    N, C_in, H, W,  # Input dimensions
    C_out, K_h, K_w,  # Output and kernel dimensions
    stride_h, stride_w,  # Stride
    padding_h, padding_w,  # Padding
    dilation_h, dilation_w,  # Dilation
    H_out, W_out,  # Output spatial dimensions
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_CO: tl.constexpr,
):
    # Get program IDs for output spatial positions and output channels
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c_out = tl.program_id(3)
    
    # Check bounds
    if pid_batch >= N or pid_h >= H_out or pid_w >= W_out or pid_c_out >= C_out:
        return
    
    # Initialize accumulator
    acc = tl.zeros((1,), tl.float32)
    
    # Iterate over input channels in blocks
    for c_in_start in range(0, C_in, BLOCK_SIZE_C):
        c_in_end = tl.minimum(c_in_start + BLOCK_SIZE_C, C_in)
        
        # Iterate over kernel height
        for kh in range(K_h):
            # Calculate input h position
            in_h = pid_h * stride_h - padding_h + kh * dilation_h
            
            # Check if input h is valid
            if in_h >= 0 and in_h < H:
                # Iterate over kernel width
                for kw in range(K_w):
                    # Calculate input w position
                    in_w = pid_w * stride_w - padding_w + kw * dilation_w
                    
                    # Check if input w is valid
                    if in_w >= 0 and in_w < W:
                        # Iterate over input channels in this block
                        c_in_range = tl.arange(0, BLOCK_SIZE_C)
                        c_in_mask = c_in_range < (c_in_end - c_in_start)
                        c_in_actual = c_in_start + c_in_range
                        
                        # Load input values
                        x_offset = (pid_batch * C_in * H * W + 
                                   c_in_actual * H * W + 
                                   in_h * W + 
                                   in_w)
                        x_vals = tl.load(x_ptr + x_offset, mask=c_in_mask, other=0.0)
                        
                        # Load weight values
                        w_offset = (pid_c_out * C_in * K_h * K_w + 
                                   c_in_actual * K_h * K_w + 
                                   kh * K_w + 
                                   kw)
                        w_vals = tl.load(w_ptr + w_offset, mask=c_in_mask, other=0.0)
                        
                        # Accumulate
                        acc += tl.sum(x_vals * w_vals)
    
    # Add bias if available
    if b_ptr is not None:
        bias_val = tl.load(b_ptr + pid_c_out)
        acc += bias_val
    
    # Store result
    out_offset = (pid_batch * C_out * H_out * W_out + 
                 pid_c_out * H_out * W_out + 
                 pid_h * W_out + 
                 pid_w)
    tl.store(out_ptr + out_offset, acc.to(x_ptr.dtype.element_ty))


# Even more optimized version using shared memory tiling
@triton.jit
def conv2d_kernel_tiled(
    x_ptr,  # Input tensor pointer (N, C_in, H, W)
    w_ptr,  # Weight tensor pointer (C_out, C_in, K_h, K_w)
    b_ptr,  # Bias tensor pointer (C_out,) - can be None
    out_ptr,  # Output tensor pointer (N, C_out, H_out, W_out)
    N, C_in, H, W,  # Input dimensions
    C_out, K_h, K_w,  # Output and kernel dimensions
    stride_h, stride_w,  # Stride
    padding_h, padding_w,  # Padding
    dilation_h, dilation_w,  # Dilation
    H_out, W_out,  # Output spatial dimensions
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_CO: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_h_block = tl.program_id(1)
    pid_w_block = tl.program_id(2)
    pid_c_out_block = tl.program_id(3)
    
    # Calculate output tile boundaries
    out_h_start = pid_h_block * BLOCK_SIZE_H
    out_w_start = pid_w_block * BLOCK_SIZE_W
    c_out_start = pid_c_out_block * BLOCK_SIZE_CO
    
    # Check bounds
    if pid_batch >= N or out_h_start >= H_out or out_w_start >= W_out or c_out_start >= C_out:
        return
    
    # Create ranges for the tile
    out_h_range = out_h_start + tl.arange(0, BLOCK_SIZE_H)
    out_w_range = out_w_start + tl.arange(0, BLOCK_SIZE_W)
    c_out_range = c_out_start + tl.arange(0, BLOCK_SIZE_CO)
    
    # Create masks
    h_mask = out_h_range < H_out
    w_mask = out_w_range < W_out
    c_out_mask = c_out_range < C_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_CO), tl.float32)
    
    # Iterate over input channels
    for c_in in range(C_in):
        # Iterate over kernel height
        for kh in range(K_h):
            # Calculate input h position
            in_h = out_h_range[:, None, None] * stride_h - padding_h + kh * dilation_h
            
            # Check if input h is valid
            h_valid = (in_h >= 0) & (in_h < H)
            
            # Iterate over kernel width
            for kw in range(K_w):
                # Calculate input w position
                in_w = out_w_range[None, :, None] * stride_w - padding_w + kw * dilation_w
                
                # Check if input w is valid
                w_valid = (in_w >= 0) & (in_w < W)
                
                # Combined valid mask
                valid_mask = h_valid & w_valid
                
                # Load input values with proper broadcasting
                x_offset = (pid_batch * C_in * H * W + 
                           c_in * H * W + 
                           in_h * W + 
                           in_w)
                
                # We need to handle the indexing carefully
                # For simplicity, we'll use a more straightforward approach
                # that works with Triton's indexing
    
    # Actually, let's use a more practical implementation
    # that processes one output element at a time but with good tiling
    
    # For the final implementation, we'll use a simpler but effective approach
    # that works well for the given asymmetric input dimensions
    pass  # We'll implement the working version below


# Practical implementation that works well
def triton_conv2d(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    bias: torch.Tensor = None,
    stride: int = 1, 
    padding: int = 0, 
    dilation: int = 1, 
    groups: int = 1
) -> torch.Tensor:
    """
    Triton-based 2D convolution implementation optimized for asymmetric inputs.
    
    Args:
        x: Input tensor of shape (N, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in // groups, K_h, K_w)
        bias: Optional bias tensor of shape (C_out,)
        stride, padding, dilation, groups: Convolution parameters
    
    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    N, C_in, H, W = x.shape
    C_out, C_in_group, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Set up grid dimensions for parallel processing
    # Using a 4D grid: (batch, output_h, output_w, output_channel)
    # For better performance on asymmetric inputs, we'll use different block sizes
    
    BLOCK_H = 8  # Block size for height
    BLOCK_W = 16  # Block size for width (larger since width is bigger)
    BLOCK_C_OUT = 8  # Block size for output channels
    
    # Calculate grid dimensions
    grid_h = (H_out + BLOCK_H - 1) // BLOCK_H
    grid_w = (W_out + BLOCK_W - 1) // BLOCK_W
    grid_c_out = (C_out + BLOCK_C_OUT - 1) // BLOCK_C_OUT
    
    grid = (N, grid_h, grid_w, grid_c_out)
    
    # Define kernel with proper tiling
    @triton.jit
    def conv2d_kernel_final(
        x_ptr, w_ptr, b_ptr, out_ptr,
        N, C_in, H, W, C_out, K_h, K_w,
        stride, padding, dilation,
        H_out, W_out,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_C_OUT: tl.constexpr,
    ):
        # Program IDs
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_w = tl.program_id(2)
        pid_c_out = tl.program_id(3)
        
        # Calculate tile boundaries
        out_h_start = pid_h * BLOCK_H
        out_w_start = pid_w * BLOCK_W
        c_out_start = pid_c_out * BLOCK_C_OUT
        
        # Create ranges
        out_h_range = out_h_start + tl.arange(0, BLOCK_H)
        out_w_range = out_w_start + tl.arange(0, BLOCK_W)
        c_out_range = c_out_start + tl.arange(0, BLOCK_C_OUT)
        
        # Create masks
        h_mask = out_h_range < H_out
        w_mask = out_w_range < W_out
        c_out_mask = c_out_range < C_out
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C_OUT), tl.float32)
        
        # Iterate over input channels
        for c_in in range(C_in):
            # Iterate over kernel height
            for kh in range(K_h):
                # Calculate input h positions
                in_h = out_h_range[:, None, None] * stride - padding + kh * dilation
                h_valid = (in_h >= 0) & (in_h < H)
                
                # Iterate over kernel width
                for kw in range(K_w):
                    # Calculate input w positions
                    in_w = out_w_range[None, :, None] * stride - padding + kw * dilation
                    w_valid = (in_w >= 0) & (in_w < W)
                    
                    valid_mask = h_valid & w_valid
                    
                    # Calculate offsets for input
                    # We need to handle the indexing carefully
                    # For simplicity, use a more direct approach
                    
                    # Process each valid position
                    for i_h in range(BLOCK_H):
                        for i_w in range(BLOCK_W):
                            for i_c_out in range(BLOCK_C_OUT):
                                # Check bounds
                                current_h = out_h_start + i_h
                                current_w = out_w_start + i_w
                                current_c_out = c_out_start + i_c_out
                                
                                if (current_h >= H_out or current_w >= W_out or 
                                    current_c_out >= C_out):
                                    continue
                                
                                # Calculate input position
                                input_h = current_h * stride - padding + kh * dilation
                                input_w = current_w * stride - padding + kw * dilation
                                
                                if (input_h >= 0 and input_h < H and 
                                    input_w >= 0 and input_w < W):
                                    # Calculate offsets
                                    x_offset = (pid_n * C_in * H * W + 
                                               c_in * H * W + 
                                               input_h * W + 
                                               input_w)
                                    w_offset = (current_c_out * C_in * K_h * K_w + 
                                               c_in * K_h * K_w + 
                                               kh * K_w + 
                                               kw)
                                    
                                    x_val = tl.load(x_ptr + x_offset)
                                    w_val = tl.load(w_ptr + w_offset)
                                    
                                    acc[i_h, i_w, i_c_out] += x_val * w_val
        
        # Add bias if available
        if b_ptr is not None:
            for i_c_out in range(BLOCK_C_OUT):
                if c_out_start + i_c_out < C_out:
                    bias_val = tl.load(b_ptr + c_out_start + i_c_out)
                    acc[:, :, i_c_out] += bias_val
        
        # Store results
        for i_h in range(BLOCK_H):
            for i_w in range(BLOCK_W):
                for i_c_out in range(BLOCK_C_OUT):
                    current_h = out_h_start + i_h
                    current_w = out_w_start + i_w
                    current_c_out = c_out_start + i_c_out
                    
                    if (current_h < H_out and current_w < W_out and 
                        current_c_out < C_out):
                        out_offset = (pid_n * C_out * H_out * W_out + 
                                     current_c_out * H_out * W_out + 
                                     current_h * W_out + 
                                     current_w)
                        tl.store(out_ptr + out_offset, acc[i_h, i_w, i_c_out])
    
    # Launch kernel
    conv2d_kernel_final[grid](
        x, weight, bias, out,
        N, C_in, H, W, C_out, K_h, K_w,
        stride, padding, dilation,
        H_out, W_out,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_C_OUT=BLOCK_C_OUT,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernels for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the same parameters as the original model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias tensors
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the optimized Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


# Alternative simpler implementation that's more reliable
def triton_conv2d_simple(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Simpler but effective Triton convolution implementation.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    N, C_in, H, W = x.shape
    C_out, C_in_group, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - dilation * (K_h - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Simple kernel that processes each output element
    @triton.jit
    def conv2d_simple_kernel(
        x_ptr, w_ptr, b_ptr, out_ptr,
        N, C_in, H, W, C_out, K_h, K_w,
        stride, padding, dilation,
        H_out, W_out,
    ):
        # 4D grid: (batch, h, w, channel)
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_w = tl.program_id(2)
        pid_c = tl.program_id(3)
        
        # Check bounds
        if pid_n >= N or pid_h >= H_out or pid_w >= W_out or pid_c >= C_out:
            return
        
        # Compute convolution for this output element
        val = 0.0
        
        for c_in in range(C_in):
            for kh in range(K_h):
                for kw in range(K_w):
                    # Calculate input position
                    in_h = pid_h * stride - padding + kh * dilation
                    in_w = pid_w * stride - padding + kw * dilation
                    
                    if 0 <= in_h < H and 0 <= in_w < W:
                        # Calculate offsets
                        x_offset = (pid_n * C_in * H * W + 
                                   c_in * H * W + 
                                   in_h * W + 
                                   in_w)
                        w_offset = (pid_c * C_in * K_h * K_w + 
                                   c_in * K_h * K_w + 
                                   kh * K_w + 
                                   kw)
                        
                        x_val = tl.load(x_ptr + x_offset)
                        w_val = tl.load(w_ptr + w_offset)
                        val += x_val * w_val
        
        # Add bias if available
        if b_ptr is not None:
            val += tl.load(b_ptr + pid_c)
        
        # Store result
        out_offset = (pid_n * C_out * H_out * W_out + 
                     pid_c * H_out * W_out + 
                     pid_h * W_out + 
                     pid_w)
        tl.store(out_ptr + out_offset, val)
    
    # Set grid dimensions
    grid = (N, H_out, W_out, C_out)
    
    # Launch kernel
    conv2d_simple_kernel[grid](
        x, weight, bias, out,
        N, C_in, H, W, C_out, K_h, K_w,
        stride, padding, dilation,
        H_out, W_out
    )
    
    return out


class ModelNewSimple(nn.Module):
    """
    Optimized version of Model using custom Triton kernels for 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNewSimple, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight and bias tensors
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the optimized Triton kernel.
        """
        return triton_conv2d_simple(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )