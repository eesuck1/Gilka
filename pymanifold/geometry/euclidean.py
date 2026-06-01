import torch

from .manifold import Manifold


class Euclidean(Manifold):
    def __init__(self, ndim: int = 1):
        super().__init__(ndim=ndim)

    def proj_x(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def proj_v(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        v_dot = torch.linalg.vecdot(x, v, dim=-1).unsqueeze(-1)

        return v - v_dot * x

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return x + v

    def retract(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return x + v

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.norm(None, u - v)

    def log_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v - x

    def transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v

    def vec_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v
