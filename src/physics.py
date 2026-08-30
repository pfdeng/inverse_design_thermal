"""Finite-difference Navier-Stokes residuals for the physics-informed loss.

Fields are (B, gy, gx): axis 1 = y (rows), axis 2 = x (cols).
Residuals are evaluated on physical (denormalised) fields, over an eroded
fluid mask so that central-difference stencils stay inside the domain.
"""
import torch
import torch.nn.functional as F
from .config import DOMAIN, GRID

DX = DOMAIN.L / (GRID.gx - 1)
DY = (DOMAIN.ybox_max - DOMAIN.ybox_min) / (GRID.gy - 1)


def _ddx(f):
    d = torch.zeros_like(f)
    d[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * DX)
    return d


def _ddy(f):
    d = torch.zeros_like(f)
    d[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * DY)
    return d


def _d2x(f):
    d = torch.zeros_like(f)
    d[:, :, 1:-1] = (f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]) / DX ** 2
    return d


def _d2y(f):
    d = torch.zeros_like(f)
    d[:, 1:-1, :] = (f[:, 2:, :] - 2 * f[:, 1:-1, :] + f[:, :-2, :]) / DY ** 2
    return d


def erode_mask(mask, k=3):
    """Keep only cells whose kxk neighbourhood is fully fluid."""
    m = mask.unsqueeze(1)
    e = -F.max_pool2d(-m, k, stride=1, padding=k // 2)
    return e.squeeze(1)


def ns_residuals(fields, nu, mask):
    """fields: (B,4,gy,gx) = [ux,uy,p,T] physical. nu: (B,).
    Returns (continuity, momentum_x, momentum_y) residual maps (B,gy,gx)."""
    ux, uy, p = fields[:, 0], fields[:, 1], fields[:, 2]
    nu = nu.view(-1, 1, 1)

    ux_x, ux_y = _ddx(ux), _ddy(ux)
    uy_x, uy_y = _ddx(uy), _ddy(uy)
    p_x, p_y = _ddx(p), _ddy(p)
    lap_ux = _d2x(ux) + _d2y(ux)
    lap_uy = _d2x(uy) + _d2y(uy)

    cont = ux_x + uy_y
    mom_x = ux * ux_x + uy * ux_y + p_x - nu * lap_ux
    mom_y = ux * uy_x + uy * uy_y + p_y - nu * lap_uy
    return cont, mom_x, mom_y


def physics_loss(fields, nu, mask):
    em = erode_mask(mask, k=3)
    denom = em.sum() + 1e-6
    cont, mx, my = ns_residuals(fields, nu, mask)
    l_cont = (cont ** 2 * em).sum() / denom
    l_mom = ((mx ** 2 + my ** 2) * em).sum() / denom
    return l_cont, l_mom
