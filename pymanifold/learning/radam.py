from typing import Callable

import torch
import torch.optim as optim

from ..geometry.manifold import Manifold


class RAdam(optim.Optimizer):
    def __init__(self, params, manifold: Manifold, lr: float = 3e-4, betas: tuple[float, float] = (0.9, 0.99), eps: float = 1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, manifold=manifold, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Performs the optimization step on given Manifolds.
        """

        for group in self.param_groups:
            lr = group["lr"]
            manifold = group["manifold"]

            for params in group["params"]:
                if params.grad is None:
                    continue

                state = self.state[params]

                if len(state) == 0:
                    state["t"] = 0

                    state["tau"] = torch.zeros_like(params, dtype=params.dtype, device=params.device)
                    state["m"] = torch.zeros_like(params, dtype=params.dtype, device=params.device)
                    state["v"] = torch.tensor(0.0, dtype=params.dtype, device=params.device)

                beta_1, beta_2 = group["betas"]
                eps = group["eps"]

                state["t"] = state["t"] + 1
                prev_m, prev_v, prev_tau = state["m"], state["v"], state["tau"]

                grad = manifold.proj_v(params, params.grad)

                tau = manifold.transport(params, prev_tau, prev_m)
                m = beta_1 * tau + (1.0 - beta_1) * grad
