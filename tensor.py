import numpy as np
from function import *

class Tensor:
    def __init__(self, data, requires_grad: bool = True, ctx=None):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self.ctx = ctx
    
    def __repr__(self):
        data_str = np.array2string(self.data, precision=4, suppress_small=True, prefix=' ' * 8)
        grad_str = np.array2string(self.grad, precision=4, suppress_small=True, prefix=' ' * 8)
        
        return (f"Tensor(shape={self.data.shape}\n"
                f" data: {data_str}\n"
                f" grad: {grad_str}")
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        return Add.apply(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return self.__mul__(-1)
    
    def __sub__(self, other):
        return self.__add__(-other)
    
    def __rsub__(self, other):
        return (-self).__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        return Mul.apply(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        return Pow.apply(self, other)
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        return MatMul.apply(self, other)
    
    def sum(self, axis=None, keepdims=False):
        return Sum.apply(self, axis, keepdims)
   
    def backward(self):
        if self.data.ndim != 0 or self.data.size != 1:
            raise RuntimeError("backward() should be called only on scalar-valued Tensors (e.g., loss).")

        self.grad = np.ones_like(self.data) 

        # Build topological order
        visited = set()
        sorted_tensors = []

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v.ctx and v.ctx.inputs:
                    for parent in v.ctx.inputs:
                        build_topo(parent)
                sorted_tensors.append(v)

        build_topo(self)

        # Reverse topological order and propagate gradients
        for tensor in reversed(sorted_tensors):
            if tensor.ctx and tensor.ctx.grad_fn:
                # Call backward on the grad_fn with upstream gradient
                input_grads = tensor.ctx.grad_fn.backward(tensor.ctx,tensor.grad)
                if not isinstance(input_grads, (list, tuple)):
                    input_grads = (input_grads,)

                for parent, grad in zip(tensor.ctx.inputs, input_grads):
                    if parent.requires_grad and grad is not None:
                        if grad.shape != parent.grad.shape:
                            # Handle broadcasting or reshaping
                            grad = grad.reshape(parent.grad.shape)
                        parent.grad += grad
   