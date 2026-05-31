import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def pointwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr
):
    # Get program IDs
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate grid dimensions
    grid_m = (out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_n = (batch_size * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Calculate which group this program belongs to
    group_id = pid // GROUP_SIZE_M
    group_size_m = min(GROUP_SIZE_M, grid_m - group_id * GROUP_SIZE_M)
    
    # Calculate the starting positions for this thread block
    block_start_m = group_id * GROUP_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE
    
    # Create masks for valid elements
    m_offsets = block_start_m + tl.arange(0, BLOCK_SIZE)
    n_offsets = block_start_n + tl.arange(0, BLOCK_SIZE)
    
    # Ensure we don't go out of bounds
    m_mask = m_offsets < out_channels
    n_mask = n_offsets < (batch_size * height * width)
    
    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over input channels (for matrix multiplication)
    for c in range(0, in_channels, BLOCK_SIZE):
        # Load weights: (out_channels, in_channels) -> (out_channels, BLOCK_SIZE)
        w_offset = c * out_channels + m_offsets[:, None]
        w_mask = m_mask[:, None] & (c + tl.arange(0, BLOCK_SIZE)[None, :] < in_channels)
        w = tl.load(weight_ptr + w_offset, mask=w_mask, other=0.0)
        
        # Load input: (batch_size, in_channels, height, width) -> (batch_size * height * width, BLOCK_SIZE)
        # This requires careful indexing since we're processing one pixel at a time
        if c == 0:
            # For first channel, we load input data
            input_idx = n_offsets
            input_mask = n_mask
            input_data = tl.load(input_ptr + input_idx, mask=input_mask, other=0.0)
            
            # Reshape input for computation
            input_data = input_data[:, None]
            # Broadcast input across output channels
            acc += tl.dot(input_data, w)
        else:
            # For subsequent channels, accumulate
            input_idx = n_offsets + c * (batch_size * height * width)
            input_mask = n_mask
            input_data = tl.load(input_ptr + input_idx, mask=input_mask, other=0.0)
            input_data = input_data[:, None]
            acc += tl.dot(input_data, w)
    
    # Write result back to global memory
    output_offset = n_offsets + m_offsets[:, None] * (batch_size * height * width)
    output_mask = n_mask[:, None] & m_mask[None, :]
    tl.store(output_ptr + output_offset, acc, mask=output_mask)

@triton.jit
def fused_matmul_relu_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate grid dimensions
    grid_size = (batch_size * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offset for this thread
    offset = pid * BLOCK_SIZE
    
    # Create mask for valid elements
    mask = offset + tl.arange(0, BLOCK_SIZE) < batch_size * height * width
    
    # Load input data
    input_data = tl.load(input_ptr + offset, mask=mask, other=0.0)
    
    # Load weight matrix (transposed)
    # Assuming weight is (out_channels, in_channels)
    # We'll compute output per row of weight
    for i in range(0, out_channels, BLOCK_SIZE):
        # Load weight slice
        w_slice = weight_ptr + i * in_channels + tl.arange(0, BLOCK_SIZE)[:, None]
        w_mask = (i + tl.arange(0, BLOCK_SIZE)[:, None]) < out_channels
        
        # Perform matmul operation
        # Note: This is a simplified version; full implementation would require more complex indexing
        if i == 0:
            result = tl.dot(input_data[:, None], weight_ptr[i * in_channels:i * in_channels + BLOCK_SIZE])
        else:
            result += tl.dot(input_data[:, None], weight_ptr[i * in_channels:i * in_channels + BLOCK_SIZE])
    
    # Apply ReLU activation
    result = tl.maximum(result, 0.0)
    
    # Store result
    tl.store(output_ptr + offset, result, mask=mask)

def triton_pointwise_conv2d(input_tensor, weight, bias=None):
    """
    Custom Triton kernel for pointwise 2D convolution
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda, "Weight tensor must be on CUDA"
    
    # Get tensor dimensions
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Flatten input for easier processing
    input_flat = input_tensor.view(-1, in_channels)  # (batch_size*height*width, in_channels)
    output_flat = torch.empty(batch_size, out_channels, height, width, device='cuda', dtype=torch.float32)
    
    # Prepare input for kernel
    input_ptr = input_flat.data_ptr()
    weight_ptr = weight.data_ptr()
    output_ptr = output_flat.data_ptr()
    
    # Define block size and group sizes
    BLOCK_SIZE = 128
    GROUP_SIZE_M = 8
    GROUP_SIZE_N = 8
    
    # Calculate grid dimensions
    grid_m = (out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_n = (batch_size * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_m * GROUP_SIZE_M, grid_n)
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        input_ptr,
        weight_ptr,
        output_ptr,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE,
        GROUP_SIZE_M,
        GROUP_SIZE_N
    )
    
    return output_flat

class ModelNew(nn.Module):
    """
    Optimized pointwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        
        # Initialize weight and bias tensors
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Convert to float32 if not already
        if x.dtype != torch.float32:
            x = x.float()
        
        # Use Triton kernel for pointwise convolution
        result = triton_pointwise_conv2d(x, self.weight, self.bias)
        
        return result

# Simplified version focusing on most critical optimization
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create block pointers
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create pointers for matrices
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M and offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k and offs_bn[None, :] < N, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Apply activation function
    if ACTIVATION == "relu":
        accumulator = tl.maximum(accumulator, 0.0)
    
    # Compute output pointer
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    # Store result
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul_tiled(A, B, activation="none"):
    """
    Efficient matrix multiplication using Triton
    """
    assert A.is_cuda and B.is_cuda, "Both tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Both tensors must be float32"
    
    M, K = A.shape
    K, N = B.shape
    
    # Allocate output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    # Define block size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    grid = (tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        activation
    )
    
    return C

# Final optimized implementation
class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        
        # Initialize weight and bias tensors
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, device='cuda'))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device='cuda'))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized pointwise 2D convolution using fused operations.
        """
        batch_size, _, height, width = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.view(batch_size, self.in_channels, -1)  # (batch_size, in_channels, height*width)
        
        # Transpose to make it (batch_size, height*width, in_channels)
        x_transposed = x_flat.transpose(1, 2)  # (batch_size, height*width, in_channels)
        
        # Apply linear transformation using Triton kernel
        # We use matrix multiplication where each sample is processed independently
        # Weight shape: (out_channels, in_channels)
        # Input shape: (batch_size * height * width, in_channels)
        # Result should be (batch_size * height * width, out_channels)
        
        # Reshape input for matmul: (batch_size * height * width, in_channels)
        input_reshaped = x_transposed.reshape(-1, self.in_channels)
        
        # Matrix multiplication using Triton
        output_reshaped = triton_matmul_tiled(input_reshaped, self.weight.t())
        
        # Add bias if present
        if self.bias is not None:
            output_reshaped += self.bias
            
        # Reshape back to original format
        output = output_reshaped.view(batch_size, height, width, self.out_channels)
        output = output.permute(0, 3, 1, 2)  # (batch_size, out_channels, height, width)
        
        return output