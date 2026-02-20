from tensor import Tensor
from typing import List

# SGD with Momentum

class SGD:
    def __init__(self, params: List[Tensor], lr: float = 0.01, beta: float = 0.9):
        if not isinstance(params, list) or not all(isinstance(p, Tensor) for p in params):
            raise TypeError("params must be a list of Tensor objects") 
        self.params = params 
        self.lr = lr 
        # Momentum
        self.beta = beta 
        self.velocity = [p.data * 0 for p in self.params] 

    def step(self): 
        for i, p in enumerate(self.params): 
            v = self.velocity[i] 
            v = self.beta * v + (1 - self.beta) * p.grad 
            p.data -= self.lr * v 
            self.velocity[i] = v 
        
    def zero_grad(self): 
        for p in self.params: 
            p.grad *= 0

class ADAM:
    def __init__(self, params: List[Tensor], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        if not isinstance(params, list) or not all(isinstance(p, Tensor) for p in params):
            raise TypeError("`params` must be a list of `Tensor` objects")

        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = [p.data.copy() * 0 for p in self.params]  # first moment
        self.v = [p.data.copy() * 0 for p in self.params]  # second moment
        self.t = 0  # timestep

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # parameter update
            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad *= 0

