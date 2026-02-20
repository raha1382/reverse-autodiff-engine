import numpy as np

__all__ = ["Add", "Mul", "Pow", "Log", "Sum", "MatMul", "ReLU_function", "Sigmoid", "Softmax", "CrossEntropyWithSoftmax"]

class Context:
    def __init__(self, grad_fn, *inputs):
        self.grad_fn = grad_fn
        self.inputs = inputs
            
class Function:
    @staticmethod
    def forward(ctx, *args): ...
    
    @staticmethod
    def backward(ctx, upstream_grad): ...

    @classmethod
    def apply(cls, *args):
        from tensor import Tensor
        inputs = [a for a in args if isinstance(a, Tensor)]
        ctx = Context(cls, *inputs)
        
        ctx.needs_input_grad = tuple(t.requires_grad for t in inputs)
        requires_grad = any(ctx.needs_input_grad)
        
        output_data = cls.forward(ctx, *args)
        ctx = ctx if requires_grad else None
        
        output_tensor = Tensor(output_data, requires_grad, ctx)
        
        return output_tensor

def unbroadcast_to(a, shape):
    while len(a.shape) > len(shape):
        a = a.sum(axis=0)
    
    for i, dim in enumerate(shape):
        if dim == 1:
            a = a.sum(axis=i, keepdims=True)

    return a

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b = b
        return a.data + b.data
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = unbroadcast_to(upstream_grad, ctx.a.data.shape) if ctx.needs_input_grad[0] else None
        grad_b = unbroadcast_to(upstream_grad, ctx.b.data.shape) if ctx.needs_input_grad[1] else None
        
        return (grad_a, grad_b)
        
class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b = b
        return a.data * b.data
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = unbroadcast_to(ctx.b * upstream_grad, ctx.a.data.shape) if ctx.needs_input_grad[0] else None
        grad_b = unbroadcast_to(ctx.a * upstream_grad, ctx.b.data.shape) if ctx.needs_input_grad[1] else None
        
        return (grad_a, grad_b)

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b = b
        return a.data @ b.data
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = upstream_grad @ ctx.b.data.T if ctx.needs_input_grad[0] else None
        grad_b = ctx.a.data.T @ upstream_grad if ctx.needs_input_grad[1] else None
        
        return (grad_a, grad_b)
            
class Pow(Function):
    @staticmethod
    def forward(ctx, base, exp):
        ctx.base = base
        ctx.exp = exp
        ctx.output = np.pow(base.data, exp.data)
        return ctx.output
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_base = ctx.exp.data * np.pow(ctx.base.data, ctx.exp.data - 1) * upstream_grad if ctx.needs_input_grad[0] else None
        grad_exp = np.log(ctx.base.data) * ctx.output * upstream_grad if ctx.needs_input_grad[1] else None
        
        return (grad_base, grad_exp)

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.a = a
        return np.log(a.data)
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = 1/ctx.a.data * upstream_grad if ctx.needs_input_grad[0] else None
        return (grad_a,)
        
class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        ctx.a = a
        return np.sum(a.data, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = np.broadcast_to(upstream_grad, ctx.a.data.shape) if ctx.needs_input_grad[0] else None
        return (grad_a,)

class ReLU_function(Function):
    @staticmethod
    def forward(ctx, a): 
        ctx.a = a
        return np.maximum(0, a.data) 
    
    @staticmethod
    def backward(ctx, upstream_grad):
        
        mask = (ctx.a.data > 0).astype(ctx.a.data.dtype)
        grad_a = mask * upstream_grad if ctx.needs_input_grad[0] else None
        return (grad_a,)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a): 
        ctx.a = a
        ctx.result = 1.0 / (1.0 + np.exp(-ctx.a.data))
        return ctx.result
    
    @staticmethod
    def backward(ctx, upstream_grad): 
        grad_a = (ctx.result * (1.0 - ctx.result)) * upstream_grad if ctx.needs_input_grad[0] else None
        return (grad_a,)

class Softmax(Function):
    @staticmethod
    def forward(ctx, a, axis):
        ctx.a = a
        ctx.axis = axis
        a_stable = ctx.a.data - ctx.a.data.max(axis=axis, keepdims=True)
        exp = np.exp(a_stable)
        sum_exp = np.sum(exp, axis=axis, keepdims=True)
        ctx.result = exp / sum_exp
        return ctx.result
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = ctx.result * (upstream_grad - np.sum(upstream_grad * ctx.result, axis=ctx.axis, keepdims=True)) if ctx.needs_input_grad[0] else None
        return (grad_a,)

class CrossEntropyWithSoftmax(Function):
    @staticmethod
    def forward(ctx, y_true, y_pred):
        ctx.y_pred = y_pred
        ctx.y_true = y_true
        softmax_output = Softmax.apply(y_pred, -1).data
        ctx.softmax_output = softmax_output
        log_softmax = np.log(softmax_output + 1e-15)  # avoid log(0)
        loss_per_sample = -np.sum(y_true.data * log_softmax, axis=-1)
        ctx.batch_size = y_pred.data.shape[0]
        return np.mean(loss_per_sample)  # SCALAR
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = ((ctx.softmax_output - ctx.y_true.data) / ctx.batch_size) * upstream_grad if ctx.needs_input_grad[0] else None
        return (None, grad_a)