import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,   # Pointer to input tensor (N, C, H, W)
    weight_ptr,  # Pointer to weight tensor (O, I, KH, KW)
    output_ptr,  # Pointer to output tensor (N, O, OH, OW)
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_o, weight_stride_i, weight_stride_kh, weight_stride_kw,
    output_stride_n, output_stride_o, output_stride_h, output_stride_w,
    n_batch, n_in_channels, n_out_channels,
    input_h, input_w, output_h, output_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    BLOCK_SIZE: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    TILE_C: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_h_id = tl.program_id(1)
    out_w_id = tl.program_id(2)
    out_ch_id = tl.program_id(3)
    
    # Calculate output position
    out_h_start = out_h_id * TILE_H
    out_w_start = out_w_id * TILE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_tile(
        tl.make_block_ptr(
            input_ptr,
            shape=(n_batch, n_in_channels, input_h, input_w),
            strides=(input_stride_n, input_stride_c, input_stride_h, input_stride_w),
            offsets=(batch_id, 0, out_h_start - padding_h, out_w_start - padding_w),
            block_shape=(1, TILE_C, TILE_H + 2*padding_h, TILE_W + 2*padding_w),
            order=(0, 1, 2, 3)
        )
    )
    
    # Shared memory for weight tile
    shared_weight = tl.shared_tile(
        tl.make_block_ptr(
            weight_ptr,
            shape=(n_out_channels, n_in_channels, kernel_h, kernel_w),
            strides=(weight_stride_o, weight_stride_i, weight_stride_kh, weight_stride_kw),
            offsets=(out_ch_id, 0, 0, 0),
            block_shape=(1, TILE_C, kernel_h, kernel_w),
            order=(0, 1, 2, 3)
        )
    )
    
    # Initialize accumulator
    acc = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
    
    # Perform convolution
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Load weight
            w = tl.load(shared_weight[0, :, kh, kw])
            
            # Load input patch
            h_offset = kh
            w_offset = kw
            
            # Handle padding
            if h_offset >= padding_h and h_offset < input_h + padding_h and \
               w_offset >= padding_w and w_offset < input_w + padding_w:
                
                # Extract input patch
                input_patch = tl.load(shared_input[0, :, h_offset - padding_h : h_offset - padding_h + TILE_H, 
                                                   w_offset - padding_w : w_offset - padding_w + TILE_W])
                
                # Multiply and accumulate
                acc += tl.sum(w[:, None, None] * input_patch[None, :, :], axis=1)
    
    # Write output
    if out_h_start < output_h and out_w_start < output_w:
        # Clamp output dimensions
        actual_h = min(TILE_H, output_h - out_h_start)
        actual_w = min(TILE_W, output_w - out_w_start)
        
        output_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(n_batch, n_out_channels, output_h, output_w),
            strides=(output_stride_n, output_stride_o, output_stride_h, output_stride_w),
            offsets=(batch_id, out_ch_id, out_h_start, out_w_start),
            block_shape=(1, 1, actual_h, actual_w),
            order=(0, 1, 2, 3)
        )
        
        tl.store(output_ptr, acc[:actual_h, :actual_w])

def triton_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """
    Custom Triton implementation of 2D convolution
    """
    assert input_tensor.is_cuda and weight.is_cuda, "Input and weight must be on CUDA"
    
    # Get dimensions
    batch_size, in_channels, input_h, input_w = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Compute output dimensions
    output_h = (input_h + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) // stride[0] + 1
    output_w = (input_w + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) // stride[1] + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_h, output_w, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    TILE_H = 16
    TILE_W = 16
    TILE_C = 32
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (
        batch_size,
        (output_h + TILE_H - 1) // TILE_H,
        (output_w + TILE_W - 1) // TILE_W,
        out_channels
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, in_channels, out_channels,
        input_h, input_w, output_h, output_w,
        kernel_h, kernel_w,
        stride[0], stride[1],
        padding[0], padding[1],
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_H=TILE_H,
        TILE_W=TILE_W,
        TILE_C=TILE_C
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation)
        )

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}'