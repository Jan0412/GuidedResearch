import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    num_classes,  # Number of classes
    stride,  # Stride between batches
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch
    batch_id = tl.program_id(0)
    
    # Compute base pointer for this batch
    input_offset = batch_id * stride
    output_offset = batch_id * stride
    
    # Load the entire batch
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_classes
    
    # Load input values
    x = tl.load(input_ptr + input_offset + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability (online softmax)
    x_max = tl.max(x, axis=0)
    
    # Compute exp(x - max)
    x_exp = tl.exp(x - x_max)
    
    # Compute sum of exps
    x_sum = tl.sum(x_exp, axis=0)
    
    # Compute log(sum(exp)) for log-softmax
    log_sum = tl.log(x_sum)
    
    # Compute log-softmax: x - max - log(sum(exp(x - max)))
    log_softmax = x - x_max - log_sum
    
    # Store result
    tl.store(output_ptr + output_offset + offsets, log_softmax, mask=mask)


@triton.jit
def cross_entropy_forward_kernel(
    log_softmax_ptr,  # Pointer to log-softmax output
    target_ptr,  # Pointer to target indices
    output_ptr,  # Pointer to output loss
    batch_size,  # Number of batches
    num_classes,  # Number of classes
    stride,  # Stride between batches
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch
    batch_id = tl.program_id(0)
    
    # Compute offsets
    log_softmax_offset = batch_id * stride
    target_offset = batch_id  # targets is 1D array
    
    # Load target index for this batch
    target_idx = tl.load(target_ptr + target_offset)
    
    # Load log-softmax values
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_classes
    
    # Extract the log-softmax value for the target class
    log_p = tl.load(log_softmax_ptr + log_softmax_offset + target_idx)
    
    # Compute negative log-likelihood (cross entropy for this sample)
    loss = -log_p
    
    # Store result
    tl.store(output_ptr + batch_id, loss)


@triton.jit
def cross_entropy_backward_kernel(
    grad_output_ptr,  # Pointer to output gradients
    log_softmax_ptr,  # Pointer to log-softmax output
    target_ptr,  # Pointer to target indices
    grad_input_ptr,  # Pointer to input gradients
    batch_size,  # Number of batches
    num_classes,  # Number of classes
    stride,  # Stride between batches
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch
    batch_id = tl.program_id(0)
    
    # Compute offsets
    log_softmax_offset = batch_id * stride
    grad_input_offset = batch_id * stride
    target_offset = batch_id
    
    # Load target index and grad_output
    target_idx = tl.load(target_ptr + target_offset)
    go = tl.load(grad_output_ptr + batch_id)
    
    # Load log-softmax values
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_classes
    
    # Load log-softmax
    log_p = tl.load(log_softmax_ptr + log_softmax_offset + offsets, mask=mask, other=-float('inf'))
    
    # Compute softmax from log-softmax
    p = tl.exp(log_p)
    
    # Create one-hot encoding for target
    is_target = (offsets == target_idx).to(tl.float32)
    
    # Compute gradient: softmax - one_hot_target
    grad = p - is_target
    
    # Scale by grad_output and store
    tl.store(grad_input_ptr + grad_input_offset + offsets, grad * go, mask=mask)


def triton_cross_entropy(predictions, targets):
    """
    Triton implementation of cross entropy loss.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.dim() == 2, "Predictions must be 2D (batch_size, num_classes)"
    assert targets.dim() == 1, "Targets must be 1D (batch_size,)"
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size, num_classes = predictions.shape
    
    # Allocate output tensors
    log_softmax = torch.empty_like(predictions)
    loss = torch.empty(batch_size, dtype=predictions.dtype, device=predictions.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024  # Tunable block size
    grid = (batch_size,)
    
    # Launch softmax kernel
    softmax_kernel[grid](predictions, log_softmax, batch_size, num_classes, num_classes, BLOCK_SIZE=BLOCK_SIZE)
    
    # Launch cross entropy forward kernel
    cross_entropy_forward_kernel[grid](log_softmax, targets, loss, batch_size, num_classes, num_classes, BLOCK_SIZE=BLOCK_SIZE)
    
    # Return mean loss
    return loss.mean()


def triton_cross_entropy_backward(grad_output, log_softmax, targets, input_shape):
    """
    Triton implementation of cross entropy backward pass.
    """
    batch_size, num_classes = input_shape
    
    # Allocate gradient input tensor
    grad_input = torch.empty(input_shape, dtype=log_softmax.dtype, device=log_softmax.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    # Launch backward kernel
    cross_entropy_backward_kernel[grid](
        grad_output.contiguous(), 
        log_softmax.contiguous(), 
        targets.contiguous(), 
        grad_input, 
        batch_size, 
        num_classes, 
        num_classes, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return grad_input


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        # Save targets and input shape for backward pass
        ctx.save_for_backward(predictions, targets)
        return triton_cross_entropy(predictions, targets)
    
    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        batch_size, num_classes = predictions.shape
        
        # Compute log-softmax for backward pass
        log_softmax = torch.empty_like(predictions)
        BLOCK_SIZE = 1024
        grid = (batch_size,)
        softmax_kernel[grid](predictions, log_softmax, batch_size, num_classes, num_classes, BLOCK_SIZE=BLOCK_SIZE)
        
        # Create grad_output tensor for backward (same shape as predictions)
        grad_output_full = grad_output.new_full((batch_size,), grad_output.item() * batch_size)
        
        grad_input = triton_cross_entropy_backward(grad_output_full, log_softmax, targets, (batch_size, num_classes))
        
        return grad_input, None


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for cross entropy loss.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return CrossEntropyFunction.apply(predictions, targets)