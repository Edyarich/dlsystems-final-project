"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for w in self.params:
            grad = w.grad.data + w.data * self.weight_decay
            update_step = (1 - self.momentum) * grad

            if self.momentum > 0:
                if w in self.u:
                    self.u[w] *= self.momentum
                    self.u[w] += update_step
                else:
                    self.u[w] = update_step
                update_step = self.u[w]

            w.data -= self.lr * update_step

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        bias_correction=True,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction
        self.t = 0

        self.cache = {}
        self.momentum = {}

    def step(self):
        self.t += 1

        for w in self.params:
            grad = w.grad.data + w.data * self.weight_decay
            momentum_upd = (1 - self.beta1) * grad
            cache_upd = (1 - self.beta2) * grad**2
            
            if self.beta1 > 0:
                if w in self.momentum:
                    self.momentum[w] *= self.beta1
                    self.momentum[w] += momentum_upd
                else:
                    self.momentum[w] = momentum_upd
                momentum_upd = self.momentum[w]

            if self.beta2 > 0:
                if w in self.cache:
                    self.cache[w] *= self.beta2
                    self.cache[w] += cache_upd
                else:
                    self.cache[w] = cache_upd
                cache_upd = self.cache[w]

            if self.bias_correction:
                momentum_upd /= (1 - self.beta1**self.t)
                cache_upd /= (1 - self.beta2**self.t)

            w.data -= self.lr * momentum_upd / (cache_upd**0.5 + self.eps)
