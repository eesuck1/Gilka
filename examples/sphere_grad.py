import torch
import numpy

import matplotlib
import matplotlib.pyplot as plt

from pymanifold.geometry.sphere import Sphere
from pymanifold.learning.rsgd import RSGD

matplotlib.use("QtAgg")


def spherical_l2(sphere: Sphere, nods: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    dist = sphere.dist(nods, anchor)
    mse = torch.mean(dist ** 2)

    return mse


def spherical_l1(sphere: Sphere, nods: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    dist = sphere.dist(nods, anchor)
    mae = torch.mean(dist)

    return mae


@torch.no_grad()
def draw_points(nods: torch.Tensor, anchor: torch.Tensor, euclidian_anchor: torch.Tensor = None) -> None:
    n = 50
    phi_range = torch.linspace(0.0, 2 * torch.pi, n)
    theta_range = torch.linspace(0.0, torch.pi, n)

    phi, theta = torch.meshgrid(phi_range, theta_range, indexing="ij")

    x = 0.95 * torch.sin(theta) * torch.cos(phi)
    y = 0.95 * torch.sin(theta) * torch.sin(phi)
    z = 0.95 * torch.cos(theta)

    figure = plt.figure()
    axis = figure.add_subplot(projection="3d")

    axis.plot_surface(x, y, z, edgecolor="black", facecolor="white", rstride=5, cstride=5, alpha=0.8, zorder=1)
    axis.scatter(nods[:, 0], nods[:, 1], nods[:, 2], color="tab:orange", s=50, depthshade=False, zorder=2)
    axis.scatter(anchor[:, 0], anchor[:, 1], anchor[:, 2], color="tab:green", s=50, depthshade=False, zorder=2)

    if euclidian_anchor is not None:
        axis.scatter(euclidian_anchor[:, 0], euclidian_anchor[:, 1], euclidian_anchor[:, 2], color="tab:red", s=50, depthshade=False, zorder=2)

    axis.view_init(elev=43, azim=36)

    axis.set_aspect("equal", "box")
    axis.axis("off")

    plt.show()


def optimize_spherical_l2() -> None:
    d = 3
    n = 10
    sigma = 1.5

    nods = torch.ones(n, d)
    nods = nods + torch.randn_like(nods) * sigma

    S2 = Sphere(3)

    nods = S2.proj_x(nods)

    euclidian_anchor = S2.proj_x(nods.mean(dim=0, keepdim=True))

    anchor = torch.randn(1, d)
    anchor = S2.proj_x(anchor)
    anchor = anchor.detach().requires_grad_(True)

    # draw_points(nods, anchor)

    epochs = 1000
    lr = 1e-2

    optimizer = RSGD([anchor], S2, lr)

    for epoch in range(epochs + 1):
        optimizer.zero_grad()

        loss = spherical_l1(S2, nods, anchor)
        loss.backward()

        optimizer.step()

        if epoch % 50 == 0 and epoch != 0:
            lr = lr * 0.95

            print(f"[{epoch}/{epochs}] Mean L2 Distance: {loss.item():.4f} | New Learning Rate: {lr:.4f}")

    print(f"RSGD Mean Distance: {spherical_l1(S2, nods, anchor).item():.4f}")
    print(f"Euclidian Mean Distance: {spherical_l1(S2, nods, euclidian_anchor).item():.4f}")

    draw_points(nods, anchor, euclidian_anchor)


def main():
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    c = torch.tensor(4.0, requires_grad=True)

    d = a + b * c
    d.backward()

    print(f"a gradient: {a.grad:.3f}")
    print(f"b gradient: {b.grad:.3f}")
    print(f"c gradient: {c.grad:.3f}")

    optimize_spherical_l2()


if __name__ == '__main__':
    main()
