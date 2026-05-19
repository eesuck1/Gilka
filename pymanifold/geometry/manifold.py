import torch

from abc import ABC, abstractmethod
from .metric import Metric, Euclidian


class Manifold(ABC):
    def __init__(self, metric: Metric = None):
        super().__init__()

        self.metric = metric if metric else Euclidian()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def proj_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projecting point `x` from Embedding space on Manifold.
        """
        raise NotImplemented()

    @abstractmethod
    def proj_v(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Projecting vector `v` to tangent space of Manifold at point `x`.
        """
        raise NotImplemented()

    @abstractmethod
    def exp_map(self, x: torch.Tensor, v: torch.Tensor, t: float | torch.Tensor = 1.0) -> torch.Tensor:
        """
        Computes the exponential map `g(t)`, with `g(0) = x` and `g'(0) = u`. Default `t = 1`.
        """
        raise NotImplemented()

    @abstractmethod
    def log_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the vector `u` from tangent space of Manifold at point `x`, such that `exp_map(x, u, 1) = v`.
        """
        raise NotImplemented()

    @abstractmethod
    def transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the parallel transport of vector `v` from tangent space of Manifold at point `x` to tangent space at point `y`.
        """
        raise NotImplemented()

    @abstractmethod
    def dist(self, u: torch.Tensor, v: torch.Tensor) -> float | torch.Tensor:
        """
        Computes the geodesic distance between two points `u` and `v` on Manifold
        """
        raise NotImplemented()

    def retr(self, x: torch.Tensor, v: torch.Tensor, t: float | torch.Tensor = 1.0) -> torch.Tensor:
        """
        Computes the retraction (fast approximation of exponential map).
        """
        return self.exp_map(x, v, t)

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the inner product between two vectors `u` and `v` from tangent space of Manifold at point `x`.
        """

        return self.metric.inner(x, u, v)

    def norm(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the norm of the vector `v` from tangent space of Manifold at point `x`.
        """

        return self.metric.norm(x, v)
