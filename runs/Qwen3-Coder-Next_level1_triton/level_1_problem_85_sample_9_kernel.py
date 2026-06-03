import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor: (batch, in_channels, height, width)
    w_ptr,  # Weight tensor: (in_channels, 1, kernel_h, kernel_w)
    b_ptr,  # Bias tensor: (in_channels,) or None
    out_ptr,  # Output tensor: (batch, in_channels, out_h, out_w)
    batch_size, in_channels, out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program indices
    bc = tl.program_id(0)  # batch index * channels_per_block + channel
    bh = tl.program_id(1)  # output height block
    bw = tl.program_id(2)  # output width block
    
    # Calculate actual batch and channel indices
    batch_idx = bc // in_channels
    channel_idx = bc % in_channels
    
    # Calculate output position
    out_h_start = bh * BLOCK_SIZE_H
    out_w_start = bw * BLOCK_SIZE_W
    
    # Create ranges for output dimensions
    out_h_offsets = out_h_start + tl.arange(0, BLOCK_SIZE_H)
    out_w_offsets = out_w_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for valid output positions
    out_h_mask = out_h_offsets < out_h
    out_w_mask = out_w_offsets < out_w
    
    # Calculate input position corresponding to output position
    # For a given output position (oh, ow), the corresponding input position is:
    # in_h = oh * stride_h - padding_h + kh * dilation_h
    # in_w = ow * stride_w - padding_w + kw * dilation_w
    
    # Load kernel weights (fixed for all batch positions)
    # We'll load the entire kernel for this channel
    kernel_offsets_h = tl.arange(0, kernel_h)
    kernel_offsets_w = tl.arange(0, kernel_w)
    
    # Create meshgrid for kernel offsets
    kh_grid, kw_grid = tl.meshgrid(kernel_offsets_h, kernel_offsets_w)
    kh_grid = tl.reshape(kh_grid, (kernel_h * kernel_w,))
    kw_grid = tl.reshape(kw_grid, (kernel_h * kernel_w,))
    
    # Load kernel weights for this channel
    w_ptrs = w_ptr + channel_idx * kernel_h * kernel_w + kh_grid * kernel_w + kw_grid
    kernel_weights = tl.load(w_ptrs, mask=kh_grid < kernel_h and kw_grid < kernel_w, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over kernel positions
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input positions
            in_h_offsets = out_h_offsets * stride_h - padding_h + kh * dilation_h
            in_w_offsets = out_w_offsets * stride_w - padding_w + kw * dilation_w
            
            # Check if input positions are within bounds
            in_h_mask = (in_h_offsets >= 0) & (in_h_offsets < x_ptr.shape[2])  # height dimension
            in_w_mask = (in_w_offsets >= 0) & (in_w_offsets < x_ptr.shape[3])  # width dimension
            
            # Create 2D mask for valid positions
            mask = out_h_mask[:, None] & out_w_mask[None, :] & in_h_mask[:, None] & in_w_mask[None, :]
            
            # Load input values
            in_h_idx = in_h_offsets[:, None]
            in_w_idx = in_w_offsets[None, :]
            
            # Calculate input pointer offset
            # Input shape: (batch, in_channels, height, width)
            # Offset = batch_idx * (in_channels * height * width) + 
            #          channel_idx * (height * width) + 
            #          in_h_idx * width + in_w_idx
            input_offset = batch_idx * (in_channels * x_ptr.shape[2] * x_ptr.shape[3]) + \
                          channel_idx * (x_ptr.shape[2] * x_ptr.shape[3]) + \
                          in_h_idx * x_ptr.shape[3] + in_w_idx
            
            # Reshape input offsets for broadcasting
            input_offsets_flat = tl.reshape(input_offset, (BLOCK_SIZE_H * BLOCK_SIZE_W,))
            
            # Load input values
            x_vals = tl.load(x_ptr + input_offsets_flat, mask=tl.reshape(mask, (BLOCK_SIZE_H * BLOCK_SIZE_W,)), other=0.0)
            x_vals = tl.reshape(x_vals, (BLOCK_SIZE_H, BLOCK_SIZE_W))
            
            # Load kernel weight for this position
            w_val = kernel_weights[kh * kernel_w + kw]
            
            # Accumulate
            acc += x_vals * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + channel_idx)
        acc += bias
    
    # Store result
    out_offset = batch_idx * (in_channels * out_h * out_w) + \
                channel_idx * (out_h * out_w) + \
                out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    
    out_offset_flat = tl.reshape(out_offset, (BLOCK_SIZE_H * BLOCK_SIZE_W,))
    mask_flat = tl.reshape(mask, (BLOCK_SIZE_H * BLOCK_SIZE_W,))
    
    tl.store(out_ptr + out_offset_flat, acc.to(x_ptr.dtype.element_ty), mask=mask_flat)


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel using optimized Triton kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        
        # Initialize bias if requested
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)
            
        # Reset parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using optimized Triton kernel.
        """
        # Ensure input is contiguous and on GPU
        x = x.contiguous()
        self.weight = self.weight.contiguous()
        
        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]
        
        # Calculate output dimensions
        out_h = (height + 2 * self.padding_h - self.dilation_h * (self.kernel_size_h - 1) - 1) // self.stride_h + 1
        out_w = (width + 2 * self.padding_w - self.dilation_w * (self.kernel_size_w - 1) - 1) // self.stride_w + 1
        
        # Create output tensor
        out = torch.empty(batch_size, self.in_channels, out_h, out_w, dtype=x.dtype, device=x.device)
        
        # Configure kernel launch parameters
        BLOCK_SIZE_C = 1  # Process one channel at a time for depthwise conv
        BLOCK_SIZE_H = 8
        BLOCK_SIZE_W = 8
        
        # Grid dimensions: (batch * channels, output_h_blocks, output_w_blocks)
        grid = (
            batch_size * self.in_channels,
            (out_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
            (out_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
        )
        
        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, self.weight, self.bias, out,
            batch_size, self.in_channels, out_h, out_w,
            self.kernel_size_h, self.kernel_size_w,
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W
        )
        
        return out