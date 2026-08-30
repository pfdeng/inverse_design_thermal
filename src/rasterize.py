"""Interpolate an unstructured FEM solution onto a fixed Cartesian grid.

The surrogate model operates on images of fixed size (GRID.gy x GRID.gx).
Nodes outside the fluid domain are masked out (mask = 0, fields = 0).
"""
import numpy as np
from matplotlib.tri import Triangulation, LinearTriInterpolator
from .config import DOMAIN, GRID


def grid_coords():
    xs = np.linspace(0.0, DOMAIN.L, GRID.gx)
    ys = np.linspace(DOMAIN.ybox_min, DOMAIN.ybox_max, GRID.gy)
    XX, YY = np.meshgrid(xs, ys)          # (gy, gx)
    return xs, ys, XX, YY


def rasterize(sol):
    """Return a dict of (gy, gx) arrays: ux, uy, p, T, mask, plus X, Y."""
    xs, ys, XX, YY = grid_coords()
    coords, tris = sol["coords"], sol["tris"]
    triang = Triangulation(coords[0], coords[1], tris.T)

    out = {"X": XX.astype(np.float32), "Y": YY.astype(np.float32)}
    mask = None
    for key in ["ux", "uy", "p", "T"]:
        interp = LinearTriInterpolator(triang, sol[key])
        vi = interp(XX, YY)               # masked array, masked outside domain
        m = ~vi.mask if np.ma.isMaskedArray(vi) else np.ones_like(XX, bool)
        arr = np.where(m, np.ma.getdata(vi), 0.0).astype(np.float32)
        out[key] = arr
        mask = m if mask is None else (mask & m)
    out["mask"] = mask.astype(np.float32)
    return out
