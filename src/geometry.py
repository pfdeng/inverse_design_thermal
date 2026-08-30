"""Random 2D flow-channel geometry generation and body-fitted mesh building.

A channel is the region between a lower wall y_low(x) and an upper wall
y_up(x), x in [0, L].  This models a fixed out-of-plane thickness (2D)
duct whose 2D shape varies: contractions/expansions, bumps and wavy walls.

The mesh is a structured curvilinear triangular mesh (body fitted), which
is convenient for the scikit-fem Navier-Stokes solver.
"""
import numpy as np
from skfem import MeshTri
from .config import DOMAIN, MESH


def random_walls(rng, nx=None):
    """Return (x, y_low, y_up) sampled on nx points along the channel.

    Walls are smooth: a low-order Fourier perturbation plus a smooth
    contraction/expansion.  Inlet and outlet are kept flat & parallel so
    boundary conditions are clean.
    """
    L, H = DOMAIN.L, DOMAIN.H
    nx = nx or MESH.nx
    x = np.linspace(0.0, L, nx)
    s = x / L  # normalized 0..1

    # window that is 0 at inlet/outlet, 1 in the middle -> keeps ends flat
    win = np.sin(np.pi * s) ** 2

    # smooth contraction/expansion of the mean gap
    gap0 = H
    contraction = rng.uniform(-0.30, 0.30)          # net narrowing/widening
    mid_shift = rng.uniform(-0.15, 0.15)            # centerline shift
    gap = gap0 * (1.0 + contraction * win)

    # wavy perturbations on each wall (low frequency)
    def wall_perturb(amp_scale):
        p = np.zeros_like(x)
        n_modes = rng.integers(1, 4)
        for _ in range(n_modes):
            k = rng.integers(1, 4)
            amp = rng.uniform(-0.12, 0.12) * amp_scale
            ph = rng.uniform(0, 2 * np.pi)
            p += amp * np.sin(np.pi * k * s + ph)
        return p * win

    center = 0.5 * H + mid_shift * win
    low = center - 0.5 * gap + wall_perturb(1.0)
    up = center + 0.5 * gap + wall_perturb(1.0)

    # guarantee a minimum gap everywhere (no pinch-off)
    min_gap = 0.28 * H
    too_thin = (up - low) < min_gap
    if np.any(too_thin):
        mid = 0.5 * (up + low)
        low = np.where(too_thin, mid - 0.5 * min_gap, low)
        up = np.where(too_thin, mid + 0.5 * min_gap, up)

    return x, low, up


def build_mesh(x, y_low, y_up, ny=None):
    """Structured curvilinear triangular mesh between the two walls.

    Returns (mesh, meta) where meta carries the logical grid shape and the
    boundary node coordinates used for tagging.
    """
    nx = x.size
    ny = ny or MESH.ny
    eta = np.linspace(0.0, 1.0, ny)  # across-channel logical coord

    # node coordinates: shape (ny, nx)
    X = np.broadcast_to(x, (ny, nx)).copy()
    Y = y_low[None, :] + (y_up - y_low)[None, :] * eta[:, None]

    pts = np.vstack([X.ravel(), Y.ravel()])  # (2, ny*nx)

    def nid(j, i):
        return j * nx + i

    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = nid(j, i)
            n01 = nid(j, i + 1)
            n10 = nid(j + 1, i)
            n11 = nid(j + 1, i + 1)
            tris.append([n00, n01, n11])
            tris.append([n00, n11, n10])
    tris = np.array(tris, dtype=np.int64).T  # (3, ntri)

    mesh = MeshTri(pts, tris)
    meta = {
        "nx": nx, "ny": ny,
        "x": x, "y_low": y_low, "y_up": y_up,
        "L": float(x[-1]),
    }
    return mesh, meta
