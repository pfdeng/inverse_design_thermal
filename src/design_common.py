"""Shared differentiable machinery for inverse design.

The trained *forward* surrogate (use_dp=False) maps
    geometry + (Re, Pr, Umean)  ->  (u_x, u_y, p, T)
and is differentiable w.r.t. its input.  We build the surrogate input from
design variables in a differentiable way, so that a physical objective
(computed from the predicted fields) can be back-propagated to the design.

Convention: fields are (gy, gx), axis 0 = y (rows), axis 1 = x (cols).
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from .config import DOMAIN, GRID

DY = (DOMAIN.ybox_max - DOMAIN.ybox_min) / (GRID.gy - 1)
DX = DOMAIN.L / (GRID.gx - 1)


def grid_tensors(device):
    xs = np.linspace(0.0, DOMAIN.L, GRID.gx, dtype=np.float32)
    ys = np.linspace(DOMAIN.ybox_min, DOMAIN.ybox_max, GRID.gy, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)                    # (gy,gx)
    Xn = (XX - XX.min()) / (XX.max() - XX.min() + 1e-9)
    Yn = (YY - YY.min()) / (YY.max() - YY.min() + 1e-9)
    return {"xs": torch.tensor(xs, device=device),
            "ys": torch.tensor(ys, device=device),
            "YY": torch.tensor(YY, device=device),
            "Xn": torch.tensor(Xn, device=device),
            "Yn": torch.tensor(Yn, device=device)}


def soft_mask_from_walls(y_low, y_up, YY, eps=None):
    """Differentiable fluid occupancy in [0,1] for a channel between walls.
    y_low, y_up: (gx,) tensors.  YY: (gy,gx) tensor of y-coordinates."""
    eps = eps if eps is not None else 1.5 * DY
    lower = torch.sigmoid((YY - y_low[None, :]) / eps)
    upper = torch.sigmoid((y_up[None, :] - YY) / eps)
    return lower * upper


def wall_distance_feature(mask_soft, stats, device):
    """Training-consistent wall-distance channel: EDT of the hard mask,
    normalised with the model's stats.  Detached (helper feature)."""
    hard = (mask_soft.detach().cpu().numpy() > 0.5)
    wd = distance_transform_edt(hard).astype(np.float32)
    wd = (wd - stats["wd_mean"]) / stats["wd_std"] * hard
    return torch.tensor(wd, device=device)


def make_input(mask_soft, Re, Pr, Umean, stats, grids, device):
    """Assemble the (1, in_ch, gy, gx) surrogate input, differentiable in
    mask_soft.  Forward model uses scalars (Re, Pr, Umean)."""
    in_ch = stats.get("in_ch", 7)
    wd = wall_distance_feature(mask_soft, stats, device)
    scn = (np.array([Re, Pr, Umean], np.float32) - stats["sc_mean"]) / stats["sc_std"]
    chans = [mask_soft, grids["Xn"], grids["Yn"], wd]
    for c in range(len(scn)):
        chans.append(torch.full_like(mask_soft, float(scn[c])))
    inp = torch.stack(chans, dim=0)[None]          # (1,in_ch,gy,gx)
    assert inp.shape[1] == in_ch, (inp.shape, in_ch)
    return inp


def forward_fields(model, inp, stats, device):
    """Run surrogate, return physical (denormalised) fields as (4,gy,gx) tensor."""
    fmean = torch.tensor(stats["fmean"], device=device).view(-1, 1, 1)
    fstd = torch.tensor(stats["fstd"], device=device).view(-1, 1, 1)
    pn = model(inp)[0]
    return pn * fstd + fmean


# ---------------------------------------------------------------------------
# physical quantities of interest (differentiable), computed from fields
# ---------------------------------------------------------------------------
def outlet_bulk_temperature(fields, mask_soft, col=-2):
    """Flow-weighted mean temperature at the outlet column (heat picked up)."""
    ux, T = fields[0], fields[3]
    w = torch.relu(ux[:, col]) * mask_soft[:, col] + 1e-8
    return (w * T[:, col]).sum() / w.sum()


def pressure_drop(fields, mask_soft, col_in=1, col_out=-2):
    """Mean inlet-column pressure minus mean outlet-column pressure."""
    p = fields[2]
    win = mask_soft[:, col_in] + 1e-8
    wout = mask_soft[:, col_out] + 1e-8
    p_in = (win * p[:, col_in]).sum() / win.sum()
    p_out = (wout * p[:, col_out]).sum() / wout.sum()
    return p_in - p_out


def qoi_summary(fields, mask_soft):
    return {"T_out": float(outlet_bulk_temperature(fields, mask_soft)),
            "dp": float(pressure_drop(fields, mask_soft))}


def design_objective(T_out, dp, T0, dp0, mode="budget", beta=0.4, Rmax=1.5, lam=8.0):
    """Objective to MINIMISE (lower is better).  T0, dp0 = baseline values.

    mode="budget"  : maximise heat pick-up subject to a pumping-cost budget,
                     J = -(T_out/T0) + lam * relu(dp/dp0 - Rmax)^2.
                     -> "get as much heat as possible while Δp <= Rmax * Δp0".
    mode="pec"     : maximise thermal-hydraulic performance
                     PEC = (T_out/T0)/(dp/dp0)^(1/3)  -> J = -PEC.
    mode="weighted": J = -(T_out/T0) + beta * (dp/dp0).
    """
    r_dp = dp / dp0
    if mode == "budget":
        over = r_dp - Rmax
        over = torch.relu(over) if hasattr(over, "clamp") else max(over, 0.0)
        return -(T_out / T0) + lam * over ** 2
    if mode == "pec":
        denom = (r_dp.clamp(min=1e-3) ** (1.0 / 3.0)
                 if hasattr(r_dp, "clamp") else max(r_dp, 1e-3) ** (1 / 3))
        return -(T_out / T0) / denom
    return -(T_out / T0) + beta * r_dp
