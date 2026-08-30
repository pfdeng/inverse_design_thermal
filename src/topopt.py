"""Topology-optimisation-style (free-form) inverse design of a channel.

Unlike the parametric version (a handful of Fourier coefficients), here the
design is a *high-dimensional field*: the two wall profiles are free at every
x-station (2 * gx design variables).  We use the standard density-based
topology-optimisation machinery adapted to this geometry class:

    raw design field  ->  density (Helmholtz/Gaussian) filter  ->  bound
    projection (sigmoid) -> soft mask -> surrogate -> objective -> autodiff
    sensitivities -> gradient update (Adam), with a TV/curvature regulariser.

Objective (same heat-exchanger trade-off as the parametric version):
    J = -(T_out / T_out0) + beta * (Δp / Δp0) + gamma * TV(shape)

Scope note: the design stays a channel between an upper and a lower wall
(vertically simply-connected), so it remains inside the surrogate's training
distribution AND can be re-meshed and verified by the FEM solver.  Genuine
interior-obstacle topology change (pin-fins / islands, with holes in the
fluid domain) would additionally require obstacle-inclusive training data and
an unstructured re-mesher for FEM validation -- documented as a next step.

Run:  python -m src.topopt --beta 0.4 --steps 300
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import DOMAIN, GRID, MESH
from .predict import load_model, fem_reference
from .design_common import (grid_tensors, soft_mask_from_walls, make_input,
                            forward_fields, outlet_bulk_temperature, pressure_drop,
                            design_objective)
from .inverse_design import _np_qoi_from_fem

H = DOMAIN.H
GAP_MIN, GAP_MAX = 0.55 * H, 1.30 * H
G0 = float(np.log((1.0 - 0.55) / (1.30 - 1.0)))   # sigmoid(G0) -> nominal gap = H
C_AMP = 0.15 * H
EXTENT = [0, DOMAIN.L, DOMAIN.ybox_min, DOMAIN.ybox_max]


def gaussian_kernel(sigma, device):
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).view(1, 1, -1)


def density_filter(field, kernel):
    """1D Gaussian density filter with reflection padding (differentiable)."""
    r = kernel.shape[-1] // 2
    f = field.view(1, 1, -1)
    f = F.pad(f, (r, r), mode="reflect")
    return F.conv1d(f, kernel).view(-1)


def walls_from_fields(ra, rb, s, kernel):
    win = torch.sin(np.pi * s) ** 2
    za = density_filter(ra, kernel) * win
    zb = density_filter(rb, kernel) * win
    gap = GAP_MIN + (GAP_MAX - GAP_MIN) * torch.sigmoid(G0 + za)
    center = 0.5 * H + C_AMP * torch.tanh(zb)
    return center - 0.5 * gap, center + 0.5 * gap


def tv(field):
    return (field[1:] - field[:-1]).abs().mean()


def optimize(model, stats, device, grids, Re, Pr, Umean, beta, gamma, steps,
             sigma=6.0, lr=0.03, obj="budget", Rmax=1.5, lam=8.0):
    s = grids["xs"] / DOMAIN.L
    YY = grids["YY"]
    kernel = gaussian_kernel(sigma, device)
    gx = GRID.gx
    ra = torch.zeros(gx, device=device, requires_grad=True)
    rb = torch.zeros(gx, device=device, requires_grad=True)

    with torch.no_grad():
        yl, yu = walls_from_fields(ra, rb, s, kernel)
        m0 = soft_mask_from_walls(yl, yu, YY)
        f0 = forward_fields(model, make_input(m0, Re, Pr, Umean, stats, grids, device), stats, device)
        T0 = float(outlet_bulk_temperature(f0, m0)); dp0 = float(pressure_drop(f0, m0))
    print(f"baseline (straight): T_out={T0:.4f}  dp={dp0:.4f}  | DOF={2*gx}")

    opt = torch.optim.Adam([ra, rb], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = {"J": [], "T_out": [], "dp": []}
    best = {"J": np.inf, "ra": ra.detach().clone(), "rb": rb.detach().clone()}
    for it in range(steps):
        yl, yu = walls_from_fields(ra, rb, s, kernel)
        m = soft_mask_from_walls(yl, yu, YY)
        f = forward_fields(model, make_input(m, Re, Pr, Umean, stats, grids, device), stats, device)
        T_out = outlet_bulk_temperature(f, m)
        dp = pressure_drop(f, m)
        reg = tv(yl) + tv(yu)
        J = design_objective(T_out, dp, T0, dp0, mode=obj, beta=beta, Rmax=Rmax, lam=lam) + gamma * reg
        opt.zero_grad(); J.backward(); opt.step(); sched.step()
        jval = float(J.detach())
        tval, dval = float(T_out.detach()), float(dp.detach())
        hist["J"].append(jval); hist["T_out"].append(tval); hist["dp"].append(dval)
        if jval < best["J"]:
            best = {"J": jval, "ra": ra.detach().clone(), "rb": rb.detach().clone()}
        if it % 30 == 0 or it == steps - 1:
            print(f"  it {it:3d} | J {jval:+.4f} | T_out {tval:.4f} "
                  f"({(tval/T0-1)*100:+.1f}%) | dp {dval:.4f} ({(dval/dp0-1)*100:+.1f}%)")
    print(f"  best J = {best['J']:+.4f}")
    return best["ra"], best["rb"], (T0, dp0), hist, kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/model_fwd.pt")
    ap.add_argument("--Re", type=float, default=60.0)
    ap.add_argument("--Pr", type=float, default=1.0)
    ap.add_argument("--Umean", type=float, default=1.0)
    ap.add_argument("--obj", choices=["budget", "pec", "weighted"], default="budget")
    ap.add_argument("--Rmax", type=float, default=1.5, help="pumping budget: dp <= Rmax*dp0")
    ap.add_argument("--lam", type=float, default=8.0, help="budget penalty weight")
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sigma", type=float, default=6.0)
    ap.add_argument("--out", default="results/inverse_topopt.png")
    args = ap.parse_args()

    model, stats, device = load_model(args.model)
    grids = grid_tensors(device)
    ra, rb, (T0, dp0), hist, kernel = optimize(
        model, stats, device, grids, args.Re, args.Pr, args.Umean,
        args.beta, args.gamma, args.steps, sigma=args.sigma, obj=args.obj,
        Rmax=args.Rmax, lam=args.lam)

    s = grids["xs"] / DOMAIN.L
    yl, yu = walls_from_fields(ra, rb, s, kernel)
    yl_np, yu_np = yl.cpu().numpy(), yu.cpu().numpy()

    with torch.no_grad():
        m = soft_mask_from_walls(yl, yu, grids["YY"])
        f = forward_fields(model, make_input(m, args.Re, args.Pr, args.Umean, stats, grids, device), stats, device)
        T_sur = float(outlet_bulk_temperature(f, m)); dp_sur = float(pressure_drop(f, m))

    xs_mesh = np.linspace(0, DOMAIN.L, MESH.nx)
    xs_g = grids["xs"].cpu().numpy()
    fem = fem_reference(xs_mesh, np.interp(xs_mesh, xs_g, yl_np),
                        np.interp(xs_mesh, xs_g, yu_np), args.Re, args.Pr, args.Umean)
    fem_opt = _np_qoi_from_fem(fem) if fem else (np.nan, np.nan)
    fem_base = fem_reference(xs_mesh, np.full(MESH.nx, 0.0), np.full(MESH.nx, H),
                             args.Re, args.Pr, args.Umean)
    fem_base_q = _np_qoi_from_fem(fem_base) if fem_base else (np.nan, np.nan)

    def pec(T, d):
        return (T / fem_base_q[0]) / (d / fem_base_q[1]) ** (1 / 3)

    obj_desc = {"budget": f"maximise T_out s.t. dp <= {args.Rmax:.2f} x dp0",
                "pec": "maximise PEC", "weighted": f"-(T_out/T0)+{args.beta}*(dp/dp0)"}[args.obj]
    print("\n=== Topology-style (free-form) inverse design result ===")
    print(f"  objective: {obj_desc}  |  design DOF = {2*GRID.gx}")
    print(f"  {'':14s}{'T_out':>10s}{'dp':>10s}{'PEC':>8s}")
    print(f"  {'baseline FEM':14s}{fem_base_q[0]:10.4f}{fem_base_q[1]:10.4f}{1.000:8.3f}")
    print(f"  {'optimised SUR':14s}{T_sur:10.4f}{dp_sur:10.4f}{pec(T_sur,dp_sur):8.3f}")
    print(f"  {'optimised FEM':14s}{fem_opt[0]:10.4f}{fem_opt[1]:10.4f}{pec(*fem_opt):8.3f}")
    if fem_base_q[0] == fem_base_q[0]:
        print(f"  FEM-verified vs baseline:  T_out {(fem_opt[0]/fem_base_q[0]-1)*100:+.1f}%   "
              f"dp {(fem_opt[1]/fem_base_q[1]-1)*100:+.1f}%   PEC {(pec(*fem_opt)-1)*100:+.1f}%")
    print(f"  surrogate vs FEM on optimum:  T_out err "
          f"{abs(T_sur-fem_opt[0])/abs(fem_opt[0])*100:.1f}%  "
          f"dp err {abs(dp_sur-fem_opt[1])/abs(fem_opt[1])*100:.1f}%")

    _plot(hist, yl_np, yu_np, fem, args, T0, dp0)
    print(f"\nfigure -> {args.out}")


def _plot(hist, yl, yu, fem, args, T0, dp0):
    fig = plt.figure(figsize=(13, 8))
    xs = np.linspace(0, DOMAIN.L, GRID.gx)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(hist["J"]); ax1.set_title("objective J"); ax1.set_xlabel("iteration"); ax1.grid(alpha=.3)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(np.array(hist["T_out"]) / T0, label="T_out / base")
    ax2.plot(np.array(hist["dp"]) / dp0, label="dp / base")
    ax2.axhline(1, color="k", lw=.6); ax2.legend(); ax2.set_title("QoIs (normalised)"); ax2.grid(alpha=.3)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.fill_between(xs, yl, yu, color="#e6a68f", alpha=.6, label="optimised")
    ax3.plot([0, DOMAIN.L], [0, 0], "k--", lw=.6)
    ax3.plot([0, DOMAIN.L], [H, H], "k--", lw=.6, label="baseline walls")
    ax3.set_title("free-form channel shape"); ax3.legend(fontsize=8); ax3.set_ylim(DOMAIN.ybox_min, DOMAIN.ybox_max)
    if fem is not None:
        m = fem["mask"] > 0.5
        for j, (name, fld, cmap) in enumerate([
                ("|u| (FEM, optimised)", np.hypot(fem["ux"], fem["uy"]), "turbo"),
                ("p (FEM, optimised)", fem["p"], "coolwarm"),
                ("T (FEM, optimised)", fem["T"], "inferno")]):
            ax = fig.add_subplot(2, 3, 4 + j)
            im = ax.imshow(np.where(m, fld, np.nan), origin="lower", aspect="auto",
                           extent=EXTENT, cmap=cmap)
            ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.035)
    obj_txt = (f"max T_out s.t. dp<={args.Rmax:g}x" if args.obj == "budget"
               else ("max PEC" if args.obj == "pec" else f"beta={args.beta}"))
    fig.suptitle(f"Topology-style inverse design (2*{GRID.gx} DOF)  |  Re={args.Re:.0f} "
                 f"Pr={args.Pr:.1f} Umean={args.Umean:.1f}  |  obj: {obj_txt}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(args.out, dpi=100); plt.close()


if __name__ == "__main__":
    main()
