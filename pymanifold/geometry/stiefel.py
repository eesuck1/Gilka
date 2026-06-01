import torch

from typing import Literal

from .manifold import Manifold


class Stiefel(Manifold):
    def __init__(self):
        super().__init__(ndim=2)

    def proj_x(self, x: torch.Tensor, projection_type: Literal["qr", "svd"] = "qr") -> torch.Tensor:
        if projection_type == "qr":
            Q, R = torch.linalg.qr(x)

            D = torch.diagonal(R, dim1=-2, dim2=-1)
            P = torch.sgn(D)
            P[P == 0] = 1.0 + 0.0j if x.is_complex() else 1.0

            Q_hat = Q * P.unsqueeze(-2)

            return Q_hat
        elif projection_type == "svd":
            U, _, VT = torch.linalg.svd(x, full_matrices=False)

            return U @ VT
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

    def proj_v(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v - x @ v.mT @ x

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Exponential map for Stiefel is too computationally expensive. Use `retract` instead.")

    def retract(self, x: torch.Tensor, v: torch.Tensor, retraction_type: Literal["qr", "svd", "cayley"] = "qr") -> torch.Tensor:
        if retraction_type == "qr":
            return self.proj_x(x + v, projection_type="qr")
        elif retraction_type == "svd":
            return self.proj_x(x + v, projection_type="svd")
        elif retraction_type == "cayley":
            p = x.size(-1)

            U = torch.cat([v, x], dim=-1)
            V = torch.cat([x, -v], dim=-1)

            VTU = V.mT @ U

            I = torch.eye(2 * p, dtype=x.dtype, device=x.device)

            M = I - 0.5 * VTU
            M_inv = torch.linalg.inv(M)

            VtX = V.mT @ x

            return x + U @ M_inv @ VtX
        else:
            raise ValueError(f"Unknown retraction type: {retraction_type}")

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def log_map(self, x: torch.Tensor, v: torch.Tensor, ensure_on_sphere: bool = True) -> torch.Tensor:
        raise NotImplementedError()

    def transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Parallel transport for Stiefel is too computationally expensive. Use `vec_transport` instead.")

    def vec_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.proj_v(y, v)
