import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    # Input tensor: (batch, in_channels, D, H, W)
    x_ptr,
    # Kernel tensor: (in_channels, out_channels, kD, kH, kW)
    w_ptr,
    # Output tensor: (batch, out_channels, D_out, H_out, W_out)
    out_ptr,
    # Bias tensor (optional)
    b_ptr,
    # Dimensions
    batch_size, in_channels, out_channels,
    input_depth, input_height, input_width,
    kernel_depth, kernel_height, kernel_width,
    output_depth, output_height, output_width,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    output_pad_d, output_pad_h, output_pad_w,
    # Block sizes for tiling
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Program IDs for output spatial dimensions and channels
    pid_b = tl.program_id(0)  # batch index
    pid_cout = tl.program_id(1)  # output channel block index
    pid_d = tl.program_id(2)  # depth block index
    pid_h = tl.program_id(3)  # height block index  
    pid_w = tl.program_id(4)  # width block index

    # Calculate starting positions in output
    out_d_start = pid_d * BLOCK_SIZE_KD
    out_h_start = pid_h * BLOCK_SIZE_KH
    out_w_start = pid_w * BLOCK_SIZE_KW
    
    # Create output position ranges
    out_d_offsets = out_d_start + tl.arange(0, BLOCK_SIZE_KD)
    out_h_offsets = out_h_start + tl.arange(0, BLOCK_SIZE_KH)
    out_w_offsets = out_w_start + tl.arange(0, BLOCK_SIZE_KW)
    
    # Create masks for output bounds
    out_d_mask = out_d_offsets < output_depth
    out_h_mask = out_h_offsets < output_height
    out_w_mask = out_w_offsets < output_width
    
    # Broadcast masks for 3D output
    out_d_mask_3d = out_d_mask[:, None, None]
    out_h_mask_3d = out_h_mask[None, :, None]
    out_w_mask_3d = out_w_mask[None, None, :]
    
    # Initialize accumulator for output channels
    # We'll compute multiple output channels in parallel
    cout_offsets = pid_cout * BLOCK_SIZE_COUT + tl.arange(0, BLOCK_SIZE_COUT)
    cout_mask = cout_offsets < out_channels
    
    # Create 3D mask for output channels
    cout_mask_3d = cout_mask[:, None, None, None]
    
    # Accumulator tensor for output
    output = tl.zeros((BLOCK_SIZE_COUT, BLOCK_SIZE_KD, BLOCK_SIZE_KH, BLOCK_SIZE_KW), dtype=tl.float32)
    
    # Loop over input channels
    for cin in range(0, in_channels, BLOCK_SIZE_CIN):
        cin_offsets = cin + tl.arange(0, BLOCK_SIZE_CIN)
        cin_mask = cin_offsets < in_channels
        
        # Load input data: (batch, in_channels, D_in, H_in, W_in)
        # Calculate corresponding input positions for each output position
        # For transpose convolution: input_d = (out_d - pad_d + output_pad_d) // stride_d
        in_d_start = (out_d_start - pad_d + output_pad_d) // stride_d
        in_h_start = (out_h_start - pad_h + output_pad_h) // stride_h
        in_w_start = (out_w_start - pad_w + output_pad_w) // stride_w
        
        # Calculate input d/h/w offsets for this block
        in_d_offsets = in_d_start + tl.arange(0, BLOCK_SIZE_KD) * stride_d
        in_h_offsets = in_h_start + tl.arange(0, BLOCK_SIZE_KH) * stride_h
        in_w_offsets = in_w_start + tl.arange(0, BLOCK_SIZE_KW) * stride_w
        
        # Check which input positions are valid
        valid_in_d = (in_d_offsets >= 0) & (in_d_offsets < input_depth)
        valid_in_h = (in_h_offsets >= 0) & (in_h_offsets < input_height)
        valid_in_w = (in_w_offsets >= 0) & (in_w_offsets < input_width)
        
        # Create mask for valid positions
        valid_mask = (valid_in_d[:, None, None] & 
                     valid_in_h[None, :, None] & 
                     valid_in_w[None, None, :])
        
        # Create full mask including output bounds and valid positions
        full_mask = (out_d_mask_3d & out_h_mask_3d & out_w_mask_3d & 
                    valid_mask[:, :, :, None] & cin_mask[None, None, None, :])
        
        # Calculate actual input indices (using where for masking)
        actual_in_d = tl.where(valid_in_d, in_d_offsets, 0)
        actual_in_h = tl.where(valid_in_h, in_h_offsets, 0)
        actual_in_w = tl.where(valid_in_w, in_w_offsets, 0)
        
        # Load input block
        input_block = tl.load(
            x_ptr + pid_b * (in_channels * input_depth * input_height * input_width) +
            cin_offsets[None, None, None, :] * (input_depth * input_height * input_width) +
            actual_in_d[:, None, None, None] * (input_height * input_width) +
            actual_in_h[None, :, None, None] * (input_width) +
            actual_in_w[None, None, :, None],
            mask=full_mask,
            other=0.0
        )
        
        # Load kernel block
        # Kernel layout: (in_channels, out_channels, kD, kH, kW)
        kernel_block = tl.load(
            w_ptr + cin_offsets[:, None, None, None, :] * (out_channels * kernel_depth * kernel_height * kernel_width) +
            cout_offsets[None, :, None, None, None] * (kernel_depth * kernel_height * kernel_width) +
            tl.arange(0, kernel_depth)[None, None, :, None, None] * (kernel_height * kernel_width) +
            tl.arange(0, kernel_height)[None, None, None, :, None] * (kernel_width) +
            tl.arange(0, kernel_width)[None, None, None, None, :],
            mask=cout_mask_3d & cin_mask[:, None, None, None] & 
                (tl.arange(0, kernel_depth)[None, None, :, None, None] < kernel_depth) &
                (tl.arange(0, kernel_height)[None, None, None, :, None] < kernel_height) &
                (tl.arange(0, kernel_width)[None, None, None, None, :] < kernel_width),
            other=0.0
        )
        
        # Compute contribution: output += input * kernel
        # For transpose convolution, the kernel is applied in a transposed manner
        # We need to handle the transposed indexing correctly
        # The kernel indices correspond to where the input contributes in the output
        # out_d = in_d * stride_d + k_d - pad_d + output_pad_d
        # So k_d = out_d - in_d * stride_d + pad_d - output_pad_d
        
        # Calculate kernel indices for each output/input position
        kernel_d_offsets = out_d_start + tl.arange(0, BLOCK_SIZE_KD)[:, None] - in_d_offsets[None, :] * stride_d + pad_d - output_pad_d
        kernel_h_offsets = out_h_start + tl.arange(0, BLOCK_SIZE_KH)[:, None] - in_h_offsets[None, :] * stride_h + pad_h - output_pad_h
        kernel_w_offsets = out_w_start + tl.arange(0, BLOCK_SIZE_KW)[:, None] - in_w_offsets[None, :] * stride_w + pad_w - output_pad_w
        
        # Create kernel index tensors for broadcasting
        kernel_d_idx = kernel_d_offsets[:, :, None, None]
        kernel_h_idx = kernel_h_offsets[:, None, :, None]
        kernel_w_idx = kernel_w_offsets[:, None, None, :]
        
        # Create kernel mask
        kernel_d_mask = (kernel_d_idx >= 0) & (kernel_d_idx < kernel_depth)
        kernel_h_mask = (kernel_h_idx >= 0) & (kernel_h_idx < kernel_height)
        kernel_w_mask = (kernel_w_idx >= 0) & (kernel_w_idx < kernel_width)
        kernel_valid_mask = kernel_d_mask & kernel_h_mask & kernel_w_mask
        
        # Create kernel index tensor (batch, cin, cout, kd, kh, kw)
        kernel_indices = (kernel_d_idx * (kernel_height * kernel_width) +
                         kernel_h_idx * (kernel_width) +
                         kernel_w_idx)
        
        # Reshape for matmul-like operation
        # For simplicity, we'll use a different approach: compute directly
        # Multiply and accumulate
        expanded_input = input_block * kernel_valid_mask
        expanded_kernel = kernel_block * kernel_valid_mask
        
        # Compute output contribution using einsum-like operation
        # This is complex due to the indexing, so let's simplify
        # We'll iterate through the kernel dimensions explicitly
        
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Check if this kernel position is valid for current output/input positions
                    valid_k = (kernel_d_offsets == kd) & (kernel_h_offsets == kh) & (kernel_w_offsets == kw)
                    
                    # Extract the kernel weight for this position
                    w_val = tl.load(
                        w_ptr + cin_offsets[:, None, None, None] * (out_channels * kernel_depth * kernel_height * kernel_width) +
                        cout_offsets[None, :, None, None] * (kernel_depth * kernel_height * kernel_width) +
                        kd * (kernel_height * kernel_width) +
                        kh * (kernel_width) +
                        kw,
                        mask=cin_mask[:, None, None, None] & cout_mask[None, :, None, None] &
                             (kd < kernel_depth) & (kh < kernel_height) & (kw < kernel_width),
                        other=0.0
                    )
                    
                    # Get the input value
                    in_mask = (valid_in_d[:, None, None] & 
                              valid_in_h[None, :, None] & 
                              valid_in_w[None, None, :] &
                              (kernel_d_offsets == kd) &
                              (kernel_h_offsets == kh) &
                              (kernel_w_offsets == kw))
                    
                    if tl.sum(in_mask) > 0:
                        # Compute contribution for all output positions
                        contrib = input_block * w_val
                        output += tl.where(in_mask[:, :, :, None] & cout_mask[None, None, None, :], contrib, 0.0)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + cout_offsets, mask=cout_mask, other=0.0)
        output += bias[None, :, None, None, None]
    
    # Store output
    out_mask = cout_mask_3d & out_d_mask_3d & out_h_mask_3d & out_w_mask_3d
    tl.store(
        out_ptr + pid_b * (out_channels * output_depth * output_height * output_width) +
        cout_offsets[:, None, None, None] * (output_depth * output_height * output_width) +
        out_d_offsets[None, :, None, None] * (output_height * output_width) +
        out_h_offsets[None, None, :, None] * (output_width) +
        out_w_offsets[None, None, None, :],
        output,
        mask=out_mask
    )


def triton_conv_transpose3d(x, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), 
                           output_padding=(0, 0, 0), groups=1):
    """
    Triton implementation of ConvTranspose3d
    """
    # Extract dimensions
    batch_size, in_channels, input_depth, input_height, input_width = x.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_height + output_padding[1]
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_width + output_padding[2]
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, 
                     dtype=x.dtype, device=x.device)
    
    # Configure block sizes (tunable parameters)
    BLOCK_SIZE_COUT = min(32, out_channels)  # Output channel block size
    BLOCK_SIZE_KD = min(3, kernel_depth)     # Depth block size (small for kernel dim)
    BLOCK_SIZE_KH = min(5, kernel_height)    # Height block size
    BLOCK_SIZE_KW = min(5, kernel_width)     # Width block size
    BLOCK_SIZE_CIN = 1  # Input channel block size (1 for simplicity)
    
    # Calculate grid dimensions
    grid = lambda meta: (
        batch_size,
        (out_channels + meta["BLOCK_SIZE_COUT"] - 1) // meta["BLOCK_SIZE_COUT"],
        (output_depth + meta["BLOCK_SIZE_KD"] - 1) // meta["BLOCK_SIZE_KD"],
        (output_height + meta["BLOCK_SIZE_KH"] - 1) // meta["BLOCK_SIZE_KH"],
        (output_width + meta["BLOCK_SIZE_KW"] - 1) // meta["BLOCK_SIZE_KW"],
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        x, weight, out, bias,
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_depth, kernel_height, kernel_width,
        output_depth, output_height, output_width,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        BLOCK_SIZE_CIN=BLOCK_SIZE_CIN,
        BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
        BLOCK_SIZE_KD=BLOCK_SIZE_KD,
        BLOCK_SIZE_KH=BLOCK_SIZE_KH,
        BLOCK_SIZE_KW=BLOCK_SIZE_KW
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized Model using custom Triton kernel for transposed 3D convolution.
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
        self.bias_flag = bias
        
        # Create weight and bias parameters (same as nn.ConvTranspose3d)
        kernel_depth, kernel_width, kernel_height = kernel_size
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_depth, kernel_height, kernel_width)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, groups=self.groups
        )