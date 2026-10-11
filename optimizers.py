"""Dense eager Adam, avoiding torch.compile imports in restricted environments."""
import math
import torch


class EagerAdam:
    def __init__(self, parameters, lr=.001):
        self.parameters = list(parameters)
        self.lr, self.steps = lr, 0
        self.first = [torch.zeros_like(p) for p in self.parameters]
        self.second = [torch.zeros_like(p) for p in self.parameters]

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.steps += 1
        for parameter, first, second in zip(self.parameters, self.first, self.second):
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise ValueError('EagerAdam only supports dense gradients.')
            first.mul_(.9).add_(gradient, alpha=.1)
            second.mul_(.999).addcmul_(gradient, gradient, value=.001)
            denominator = second.sqrt()/math.sqrt(1-.999**self.steps)+1e-8
            parameter.addcdiv_(first, denominator, value=-self.lr/(1-.9**self.steps))
