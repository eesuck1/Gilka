import torch

from .metric import Metric
from .manifold import Manifold


class Sphere(Manifold):
    def __init__(self, dims: int, metric: Metric = None):
        super().__init__(metric)

        self.dims = dims

    def proj_x(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)

        return x / x_norm

    def proj_v(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        v_dot = torch.linalg.vecdot(x, v, dim=-1).unsqueeze(-1)

        return v - v_dot * x

    def exp_map(self, x: torch.Tensor, v: torch.Tensor, t: float | torch.Tensor = 1.0) -> torch.Tensor:
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        angle = v_norm * t

        gamma = torch.cos(angle) * x + torch.sin(angle) * v / v_norm

        return gamma

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dot_uv = torch.linalg.vecdot(u, v, dim=-1).unsqueeze(-1)
        dot_uv = torch.clamp(dot_uv, min=-1.0, max=1.0)

        dist = torch.arccos(dot_uv)

        return dist

    def log_map(self, x: torch.Tensor, v: torch.Tensor, ensure_on_sphere: bool = True) -> torch.Tensor:
        dot_xv = torch.linalg.vecdot(x, v, dim=-1).unsqueeze(-1)
        len_xv = torch.arccos(dot_xv)

        ort_xv = v - dot_xv * x
        ort_norm = torch.linalg.norm(ort_xv, dim=-1, keepdim=True)

        u = ort_xv * len_xv / ort_norm

        return u


    def transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if torch.allclose(x, -y, atol=1e-5):
            raise ValueError("Points are antipodic (x = -y)")

        xy = x + y
        xy_dot = torch.linalg.vecdot(x, y, dim=-1).unsqueeze(-1)
        yv_dot = torch.linalg.vecdot(y, v, dim=-1).unsqueeze(-1)

        u = v - (yv_dot / (1.0 + xy_dot)) * xy

        return u


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    import torch

    matplotlib.use("QtAgg")

    s_3 = Sphere(3)

    x0 = torch.tensor([1.0, 0.0, 0.0])
    x0 = s_3.proj_x(x0)

    v_dir = torch.tensor([0.0, 0.0, 1.0])
    v_dir = s_3.proj_v(x0, v_dir)

    w_trans = torch.tensor([0.0, 1.0, 0.0])
    w_trans = s_3.proj_v(x0, w_trans)

    t_vals = torch.linspace(0, 1, 15).unsqueeze(-1)
    geodesic_points = s_3.exp_map(x0, v_dir, t_vals)

    transported_vectors = []

    for p in geodesic_points:
        transported_vectors.append(s_3.transport(x0, p, w_trans))

    transported_vectors = torch.stack(transported_vectors)

    plane_size = 0.6
    grid = torch.linspace(-plane_size, plane_size, 10)
    a, b = torch.meshgrid(grid, grid, indexing="ij")

    X_plane = x0[0] + a * v_dir[0] + b * w_trans[0]
    Y_plane = x0[1] + a * v_dir[1] + b * w_trans[1]
    Z_plane = x0[2] + a * v_dir[2] + b * w_trans[2]

    u_test = torch.linspace(0, 2 * torch.pi, 30)
    v_test = torch.linspace(0, torch.pi, 20)

    X_sph = torch.outer(torch.cos(u_test), torch.sin(v_test)).numpy()
    Y_sph = torch.outer(torch.sin(u_test), torch.sin(v_test)).numpy()
    Z_sph = torch.outer(torch.ones_like(u_test), torch.cos(v_test)).numpy()

    fig = plt.figure(figsize=(18, 6))
    views = [(15, 45), (30, 120), (90, 0)]

    for i, (elev, azim) in enumerate(views):
        axis = fig.add_subplot(1, 3, i + 1, projection="3d")

        axis.plot_wireframe(X_sph, Y_sph, Z_sph, color="gray", alpha=0.15)
        axis.plot_surface(X_plane.numpy(), Y_plane.numpy(), Z_plane.numpy(), color="tab:cyan", alpha=0.3)

        pts = geodesic_points.numpy()
        axis.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="tab:orange", linewidth=3, label="Geodesic", zorder=4)

        axis.scatter(*x0.numpy(), color="black", s=50, zorder=5)

        vecs = transported_vectors.numpy()
        axis.quiver(pts[:, 0], pts[:, 1], pts[:, 2], vecs[:, 0], vecs[:, 1], vecs[:, 2],
                    color="tab:green", normalize=False, length=1.0, arrow_length_ratio=0.15, linewidth=2,
                    label="Transported Vector", zorder=6)

        axis.quiver(*x0.numpy(), *v_dir.numpy(),
                    color="tab:blue", normalize=False, length=1.0, arrow_length_ratio=0.15, linewidth=3,
                    label="Velocity", zorder=7)

        axis.view_init(elev=elev, azim=azim)
        axis.set_box_aspect([1, 1, 1])
        axis.set_xlim([-1, 1])
        axis.set_ylim([-1, 1])
        axis.set_zlim([-1, 1])
        axis.axis("off")

        if i == 0:
            axis.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
