import torch

from abc import ABC, abstractmethod


class Manifold(ABC):
    def __init__(self, ndim: int = 1):
        super().__init__()

        self.ndim = ndim
        self.reduce_dims = tuple(range(-ndim, 0))

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def proj_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projecting point `x` from Embedding space on Manifold.
        """
        raise NotImplementedError()

    @abstractmethod
    def proj_v(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Projecting vector `v` to tangent space of Manifold at point `x`.
        """
        raise NotImplementedError()

    @abstractmethod
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the exponential map `g(t)`, with `g(0) = x` and `g'(0) = v`.
        """
        raise NotImplementedError()

    @abstractmethod
    def retract(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the retraction map `g(t)`, with `g(0) = x` and `g'(0) = v`.
        """
        raise NotImplementedError()

    @abstractmethod
    def log_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the vector `u` from tangent space of Manifold at point `x`, such that `exp_map(x, u, 1) = v`.
        """
        raise NotImplementedError()

    @abstractmethod
    def transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the parallel transport of vector `v` from tangent space of Manifold at point `x` to tangent space at point `y`.
        """
        raise NotImplementedError()

    @abstractmethod
    def vec_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the vector transport of vector `v` from tangent space of Manifold at point `x` to tangent space at point `y`.
        """
        raise NotImplementedError()

    @abstractmethod
    def dist(self, u: torch.Tensor, v: torch.Tensor) -> float | torch.Tensor:
        """
        Computes the geodesic distance between two points `u` and `v` on Manifold
        """
        raise NotImplementedError()

    def retr(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the retraction (fast approximation of exponential map).
        """
        return self.exp_map(x, v)

    def inner(self, x: torch.Tensor | None, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the inner product between two vectors `u` and `v` from tangent space of Manifold at point `x`.
        """

        return torch.sum(u * v, dim=self.reduce_dims)

    def norm(self, x: torch.Tensor | None , v: torch.Tensor) -> torch.Tensor:
        """
        Computes the norm of the vector `v` from tangent space of Manifold at point `x`.
        """

        return torch.sqrt(torch.sum(v ** 2, dim=self.reduce_dims))

    def _broadcast(self, scalar: torch.Tensor) -> torch.Tensor:
        for _ in range(self.ndim):
            scalar = scalar.unsqueeze(-1)

        return scalar