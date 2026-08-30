"""Inference API: given an arbitrary 2D channel shape + operating conditions,
predict the thermal-flow field with the trained surrogate, and (optionally)
compare against a fresh FEM solve.

Operating conditions = fluid/regime (Re, Pr) + flow (Umean, sets flow rate)
+ pressure (dp).  These are the surrogate's scalar inputs.
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from .config import DOMAIN, GRID
from .model import UNet
from .rasterize import grid_coords
from .geometry import build_mesh
from .fem_solver import solve_case


def _mask_from_walls(x, y_low, y_up):
    """Rasterise the fluid mask on the fixed grid from wall profiles."""
    xs, ys, XX, YY = grid_coords()
    lo = np.interp(XX[0], x, y_low)          # (gx,)
    up = np.interp(XX[0], x, y_up)
    mask = (YY >= lo[None, :]) & (YY <= up[None, :])
    return mask.astype(np.float32), XX, YY


def load_model(path="results/model.pt", device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    in_ch = ckpt["stats"].get("in_ch", 8)
    model = UNet(in_ch=in_ch, out_ch=4, base=ckpt["args"]["base"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["stats"], device


def build_input(x, y_low, y_up, Re, Pr, Umean, dp, stats):
    mask, XX, YY = _mask_from_walls(x, y_low, y_up)
    Xn = (XX - XX.min()) / (XX.max() - XX.min() + 1e-9)
    Yn = (YY - YY.min()) / (YY.max() - YY.min() + 1e-9)
    wd = distance_transform_edt(mask > 0.5).astype(np.float32)
    wd = (wd - stats["wd_mean"]) / stats["wd_std"] * (mask > 0.5)

    use_dp = stats.get("use_dp", True)
    raw = [Re, Pr, Umean, dp] if use_dp else [Re, Pr, Umean]
    scn = (np.array(raw, np.float32) - stats["sc_mean"]) / stats["sc_std"]
    in_ch = stats.get("in_ch", 8)
    inp = np.zeros((1, in_ch, GRID.gy, GRID.gx), np.float32)
    inp[0, 0] = mask; inp[0, 1] = Xn; inp[0, 2] = Yn; inp[0, 3] = wd
    for c in range(len(scn)):
        inp[0, 4 + c] = scn[c]
    return torch.from_numpy(inp), mask


def predict(x, y_low, y_up, Re, Pr, Umean, dp=0.0,
            model=None, stats=None, device=None):
    """Return dict with predicted ux, uy, p, T on the fixed grid + mask.
    `dp` is ignored for a forward (use_dp=False) model."""
    if model is None:
        model, stats, device = load_model()
    inp, mask = build_input(x, y_low, y_up, Re, Pr, Umean, dp, stats)
    fmean = torch.tensor(stats["fmean"], device=device)
    fstd = torch.tensor(stats["fstd"], device=device)
    with torch.no_grad():
        pn = model(inp.to(device))
    phys = (pn * fstd.view(1, -1, 1, 1) + fmean.view(1, -1, 1, 1)).cpu().numpy()[0]
    phys *= (mask > 0.5)
    return {"ux": phys[0], "uy": phys[1], "p": phys[2], "T": phys[3], "mask": mask}


def fem_reference(x, y_low, y_up, Re, Pr, Umean):
    """Run the FEM solver on the same geometry for comparison."""
    from .rasterize import rasterize
    mesh, meta = build_mesh(x, y_low, y_up)
    sol = solve_case(mesh, meta, Re, Pr, Umean)
    if sol is None:
        return None
    g = rasterize(sol)
    return {"ux": g["ux"], "uy": g["uy"], "p": g["p"], "T": g["T"],
            "mask": g["mask"], "dp": sol["dp"], "Q": sol["Q"]}
