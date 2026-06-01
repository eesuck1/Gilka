import torch

import matplotlib
import matplotlib.pyplot as plt

from pymanifold.geometry.stiefel import Stiefel
from pymanifold.learning.optimizer import Ramsgrad

matplotlib.use("QtAgg")


def is_orthogonal(m: torch.Tensor) -> torch.Tensor:
    I = torch.eye(m.size(-2), m.size(-1), dtype=m.dtype, device=m.device)
    A = m.mT @ m

    return torch.isclose(I, A, atol=1e-4).all(dim=[-2, -1])


def main():
    V = Stiefel()

    X = torch.randn(3, 3)
    I = torch.eye(3, 3)

    Y = V.proj_x(X)

    print(is_orthogonal(Y))


if __name__ == '__main__':
    main()
