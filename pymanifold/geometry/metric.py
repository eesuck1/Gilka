import torch

from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def norm(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.inner(x, v, v))


class Euclidean(Metric):
    def __init__(self):
        super().__init__()

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vecdot(u, v, dim=-1)

    def norm(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(v, dim=-1)


if __name__ == "__main__":
    euc_norm = Euclidean()
    test_vec_1 = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]).view(1, 1, 2, 3)
    test_vec_2 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).view(1, 1, 2, 3)

    print(euc_norm.norm(test_vec_1, test_vec_1))
    print(euc_norm.inner(test_vec_1, test_vec_1, test_vec_2))