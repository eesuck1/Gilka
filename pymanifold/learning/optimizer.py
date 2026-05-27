from typing import Callable

import torch
import torch.optim as optim

from geometry.manifold import Manifold
from ..geometry.manifold import Manifold


class Ramsgrad(optim.Optimizer):
    def __init__(self, params, manifold: Manifold, lr: float = 3e-4, betas: tuple[float, float] = (0.9, 0.99),
                 eps: float = 1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, manifold=manifold, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Performs the Adam optimization step on given Manifolds.
        """

        for group in self.param_groups:
            lr = group["lr"]
            manifold = group["manifold"]

            for params in group["params"]:
                if params.grad is None:
                    continue

                state = self.state[params]

                if len(state) == 0:
                    state["time_step"] = 0

                    state["prev_mean"] = torch.zeros_like(params, dtype=params.dtype, device=params.device)
                    state["prev_norm"] = torch.tensor(0.0, dtype=params.dtype, device=params.device)

                beta_1, beta_2 = group["betas"]
                eps = group["eps"]

                state["time_step"] = state["time_step"] + 1

                time_step, prev_mean, prev_norm = state["time_step"], state["prev_mean"], state["prev_norm"]

                beta_1t = beta_1 / time_step

                grad = manifold.proj_v(params, params.grad)
                grad_norm = torch.linalg.norm(grad) ** 2

                norm = prev_norm * beta_2 + grad_norm * (1.0 - beta_2)
                norm_hat = norm if norm > prev_norm else prev_norm

                mean = prev_mean * beta_1t + grad * (1.0 - beta_1t)
                mean_hat = mean / (torch.sqrt(norm_hat) + eps)

                new_params = manifold.exp_map(params, -mean_hat, lr)

                state["prev_mean"] = manifold.transport(params, new_params, mean)
                state["prev_norm"] = norm_hat

                params.copy_(new_params)


class RSGD(optim.Optimizer):
    def __init__(self, params, manifold: Manifold, lr: float = 3e-4):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, manifold=manifold)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Performs the SGD optimization step on given Manifolds.
        """

        for group in self.param_groups:
            lr = group["lr"]
            manifold = group["manifold"]

            for params in group["params"]:
                if params.grad is None:
                    continue

                grad = manifold.proj_v(params, params.grad)
                new_params = manifold.exp_map(params, -grad, lr)
                params.copy_(new_params)
