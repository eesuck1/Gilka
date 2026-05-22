from typing import Callable

import torch
import torch.optim as optim

from ..geometry.manifold import Manifold


class RSGD(optim.Optimizer):
    def __init__(self, params, manifold: Manifold, lr: float = 3e-4):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, manifold=manifold)
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

                grad = manifold.proj_v(params, params.grad)
                new_params = manifold.exp_map(params, -grad, lr)
                params.copy_(new_params)
