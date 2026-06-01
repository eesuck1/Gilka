import torch

from .manifold import Manifold


class Sphere(Manifold):
    def __init__(self, ndim: int = 1):
        super().__init__(ndim=ndim)

    def proj_x(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self._broadcast(self.norm(None, x))

        return x / x_norm

    def proj_v(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        v_dot = self._broadcast(self.inner(x, x, v))

        return v - v_dot * x

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        v_norm = self._broadcast(self.norm(x, v))
        gamma = torch.cos(v_norm) * x + torch.sin(v_norm) * v / torch.clamp(v_norm, min=1e-7)

        return gamma

    def retract(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.proj_x(x + v)

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dot_uv = self.inner(None, u, v)
        dot_uv = torch.clamp(dot_uv, min=-1.0 + 1e-7, max=1.0 - 1e-7)

        return torch.arccos(dot_uv)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dot_xy = self._broadcast(self.inner(x, x, y))
        dot_xy = torch.clamp(dot_xy, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        len_xy = torch.arccos(dot_xy)

        ort_xy = y - dot_xy * x
        ort_norm = self._broadcast(self.norm(x, ort_xy))

        u = ort_xy * len_xy / torch.clamp(ort_norm, min=1e-7)

        return u

    def transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if torch.allclose(x, -y, atol=1e-5):
            raise ValueError("Points are antipodic (x = -y)")

        xy = x + y
        xy_dot = self._broadcast(self.inner(x, x, y))
        yv_dot = self._broadcast(self.inner(y, y, v))

        u = v - (yv_dot / (1.0 + xy_dot)) * xy

        return u

    def vec_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.proj_v(y, v)
