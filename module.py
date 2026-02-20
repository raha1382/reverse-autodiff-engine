import numpy as np
from tensor import Tensor
from function import *

class Module:
    def parameters(self):
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params

    def set_parameters(self, params):
        param_list = self.parameters()
        for i, p_data in enumerate(params):
            param_list[i].data = p_data
    
    def forward(self, *args, **kwargs): ...
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Tensor(np.random.randn(in_features,out_features) * np.sqrt(2 / in_features), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x):
        return MatMul.apply(x, self.weight) + self.bias

class ReLU(Module):
    def forward(self, x):
        return ReLU_function.apply(x)