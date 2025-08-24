import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Box3D:
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0

@dataclass
class QuantumNumbers:
    nx: int = 1
    ny: int = 1
    nz: int = 1


def psi_infinite_box(x, y, z, n: QuantumNumbers, box: Box3D):
    norm = 2.0 / np.sqrt(box.Lx * box.Ly * box.Lz)
    psi = (
        np.sin(n.nx * np.pi * x / box.Lx)
        * np.sin(n.ny * np.pi * y / box.Ly)
        * np.sin(n.nz * np.pi * z / box.Lz)
    )
    return norm * psi


def probability_density(n: QuantumNumbers, box: Box3D, N=128):
    """Düzenli 3B ızgarada |Ψ|^2 döndürür."""
    x = np.linspace(0, box.Lx, N)
    y = np.linspace(0, box.Ly, N)
    z = np.linspace(0, box.Lz, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    psi = psi_infinite_box(X, Y, Z, n, box)
    rho = np.abs(psi) ** 2
    return x, y, z, rho


def plot_slices(x, y, z, rho, title_prefix="|Ψ|^2 dilimler"):
    """xy, xz, yz düzlemlerinde orta dilimleri ayrı figürlerde gösterir."""
    ix = len(x) // 2
    iy = len(y) // 2
    iz = len(z) // 2

    plt.figure()
    plt.imshow(rho[:, :, iz].T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], aspect="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title_prefix}: z={z[iz]:.3f}")
    plt.colorbar(label="|Ψ|^2")
    plt.tight_layout()

    plt.figure()
    plt.imshow(rho[:, iy, :].T, origin="lower", extent=[x.min(), x.max(), z.min(), z.max()], aspect="auto")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"{title_prefix}: y={y[iy]:.3f}")
    plt.colorbar(label="|Ψ|^2")
    plt.tight_layout()

    plt.figure()
    plt.imshow(rho[ix, :, :].T, origin="lower", extent=[y.min(), y.max(), z.min(), z.max()], aspect="auto")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.title(f"{title_prefix}: x={x[ix]:.3f}")
    plt.colorbar(label="|Ψ|^2")
    plt.tight_layout()


def plot_isosurface_like(x, y, z, rho, percentile=90, max_points=200_000):

    thr = np.percentile(rho, percentile)
    mask = rho >= thr

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    xs, ys, zs = X[mask], Y[mask], Z[mask]

    # Çok yoğun olabilir; örnekleyelim
    if xs.size > max_points:
        idx = np.random.choice(xs.size, size=max_points, replace=False)
        xs, ys, zs = xs[idx], ys[idx], zs[idx]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=1, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"|Ψ|^2 > {percentile}. yüzdelik – nokta bulutu")
    plt.tight_layout()


def psi_superposition(x, y, z, terms, box: Box3D, t=0.0, hbar=1.0, m=1.0):

    X, Y, Z = x, y, z
    # Eğer x,y,z vektörse 3D grid yapalım
    if X.ndim == 1:
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    psi = np.zeros_like(X, dtype=complex)
    for n, c in terms:
        k2 = (n.nx / box.Lx) ** 2 + (n.ny / box.Ly) ** 2 + (n.nz / box.Lz) ** 2
        E = (np.pi ** 2) * (hbar ** 2) * k2 / (2 * m)
        phase = np.exp(-1j * E * t / hbar)
        psi += c * psi_infinite_box(X, Y, Z, n, box) * phase
    return psi


def demo():
    box = Box3D(1.0, 1.0, 1.0)
    n = QuantumNumbers(2, 3, 1)
    N = 128  

    x, y, z, rho = probability_density(n, box, N=N)

    plot_slices(x, y, z, rho, title_prefix=f"|Ψ|^2 dilimler (n=({n.nx},{n.ny},{n.nz}))")

    plot_isosurface_like(x, y, z, rho, percentile=92)

    plt.show()


if __name__ == "__main__":
    demo()
