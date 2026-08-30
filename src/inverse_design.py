"""Parametric inverse design of a thermal-flow channel.

Design variables: a few Fourier coefficients that shape the channel gap and
centreline.  We optimise them by gradient descent *through the differentiable
forward surrogate* (autodiff plays the role of the adjoint), for a
heat-exchanger objective:

    maximise outlet heat pick-up (T_out)  while penalising pumping cost (Δp)

    J = -(T_out / T_out0)  +  beta * (Δp / Δp0)      (minimised)

The optimised geometry is then re-simulated with the FEM solver to verify the
surrogate-driven design against ground truth.

Run:  python -m src.inverse_design --beta 0.4 --steps 200
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import DOMAIN, GRID, MESH
from .predict import load_model, fem_reference
from .design_common import (grid_tensors, soft_mask_from_walls, make_input,
                            forward_fields, outlet_bulk_temperature, pressure_drop,
                            design_objective)

H = DOMAIN.H
# Keep the design inside the surrogate's training distribution (channels with
# gap ~0.55H..1.3H, modest centreline shift) so predictions stay trustworthy.
GAP_MIN, GAP_MAX = 0.55 * H, 1.30 * H
G0 = float(np.log((1.0 - 0.55) / (1.30 - 1.0)))   # sigmoid(G0) -> nominal gap = H
C_AMP = 0.15 * H
EXTENT = [0, DOMAIN.L, DOMAIN.ybox_min, DOMAIN.ybox_max]


def walls_from_params(a, b, s):
    """a,b: (K,) tensors. s: (nx,) normalized x in [0,1]. Returns y_low,y_up."""
    win = torch.sin(np.pi * s) ** 2
    K = a.shape[0]
    modes = torch.stack([torch.sin(np.pi * (k + 1) * s) for k in range(K)])  # (K,nx)
    zg = win * (a[:, None] * modes).sum(0)
    zc = win * (b[:, None] * modes).sum(0)
    gap = GAP_MIN + (GAP_MAX - GAP_MIN) * torch.sigmoid(G0 + zg)
    center = 0.5 * H + C_AMP * torch.tanh(zc)
    return center - 0.5 * gap, center + 0.5 * gap


def _np_qoi_from_fem(fem):
    """Compute (T_out, dp) from a rasterised FEM solution (numpy)."""
    ux, p, T, m = fem["ux"], fem["p"], fem["T"], fem["mask"] > 0.5
    w = np.maximum(ux[:, -2], 0) * m[:, -2] + 1e-8
    T_out = float((w * T[:, -2]).sum() / w.sum())
    win = m[:, 1] + 1e-8; wout = m[:, -2] + 1e-8
    dp = float((win * p[:, 1]).sum() / win.sum() - (wout * p[:, -2]).sum() / wout.sum())
    return T_out, dp


def optimize(model, stats, device, grids, Re, Pr, Umean, beta, steps, K=4, lr=0.05,
             obj="budget", Rmax=1.5, lam=8.0):
    s = grids["xs"] / DOMAIN.L
    YY = grids["YY"]
    a = torch.zeros(K, device=device, requires_grad=True)
    b = torch.zeros(K, device=device, requires_grad=True)

    # baseline (straight channel) QoIs for normalisation
    with torch.no_grad():
        yl, yu = walls_from_params(a, b, s)
        m0 = soft_mask_from_walls(yl, yu, YY)
        f0 = forward_fields(model, make_input(m0, Re, Pr, Umean, stats, grids, device), stats, device)
        T0 = float(outlet_bulk_temperature(f0, m0)); dp0 = float(pressure_drop(f0, m0))
    print(f"baseline (straight): T_out={T0:.4f}  dp={dp0:.4f}")

    opt = torch.optim.Adam([a, b], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = {"J": [], "T_out": [], "dp": []}
    best = {"J": np.inf, "a": a.detach().clone(), "b": b.detach().clone()}
    for it in range(steps):
        yl, yu = walls_from_params(a, b, s)
        m = soft_mask_from_walls(yl, yu, YY)
        f = forward_fields(model, make_input(m, Re, Pr, Umean, stats, grids, device), stats, device)
        T_out = outlet_bulk_temperature(f, m)
        dp = pressure_drop(f, m)
        J = design_objective(T_out, dp, T0, dp0, mode=obj, beta=beta, Rmax=Rmax, lam=lam)
        opt.zero_grad(); J.backward(); opt.step(); sched.step()
        jval = float(J.detach())
        tval, dval = float(T_out.detach()), float(dp.detach())
        hist["J"].append(jval); hist["T_out"].append(tval); hist["dp"].append(dval)
        if jval < best["J"]:
            best = {"J": jval, "a": a.detach().clone(), "b": b.detach().clone()}
        if it % 20 == 0 or it == steps - 1:
            print(f"  it {it:3d} | J {jval:+.4f} | T_out {tval:.4f} "
                  f"({(tval/T0-1)*100:+.1f}%) | dp {dval:.4f} ({(dval/dp0-1)*100:+.1f}%)")
    print(f"  best J = {best['J']:+.4f}")
    return best["a"], best["b"], (T0, dp0), hist


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
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--out", default="results/inverse_parametric.png")
    args = ap.parse_args()

    model, stats, device = load_model(args.model)
    grids = grid_tensors(device)

    a, b, (T0, dp0), hist = optimize(model, stats, device, grids, args.Re, args.Pr,
                                     args.Umean, args.beta, args.steps, K=args.K,
                                     obj=args.obj, Rmax=args.Rmax, lam=args.lam)

    # extract optimised walls
    s = grids["xs"] / DOMAIN.L
    yl, yu = walls_from_params(a, b, s)
    yl_np, yu_np = yl.cpu().numpy(), yu.cpu().numpy()

    # surrogate prediction on optimised design
    with torch.no_grad():
        m = soft_mask_from_walls(yl, yu, grids["YY"])
        f = forward_fields(model, make_input(m, args.Re, args.Pr, args.Umean, stats, grids, device), stats, device)
        T_sur = float(outlet_bulk_temperature(f, m)); dp_sur = float(pressure_drop(f, m))

    # FEM validation on optimised design (sample walls at mesh resolution)
    xs_mesh = np.linspace(0, DOMAIN.L, MESH.nx)
    yl_m = np.interp(xs_mesh, grids["xs"].cpu().numpy(), yl_np)
    yu_m = np.interp(xs_mesh, grids["xs"].cpu().numpy(), yu_np)
    fem = fem_reference(xs_mesh, yl_m, yu_m, args.Re, args.Pr, args.Umean)
    fem_opt = _np_qoi_from_fem(fem) if fem else (np.nan, np.nan)

    # FEM baseline (straight channel)
    yl_b = np.full(MESH.nx, 0.5 * H - 0.5 * H); yu_b = np.full(MESH.nx, 0.5 * H + 0.5 * H)
    fem_base = fem_reference(xs_mesh, yl_b, yu_b, args.Re, args.Pr, args.Umean)
    fem_base_q = _np_qoi_from_fem(fem_base) if fem_base else (np.nan, np.nan)

    def pec(T, d):
        return (T / fem_base_q[0]) / (d / fem_base_q[1]) ** (1 / 3)

    obj_desc = {"budget": f"maximise T_out s.t. dp <= {args.Rmax:.2f} x dp0",
                "pec": "maximise PEC=(T_out/T0)/(dp/dp0)^(1/3)",
                "weighted": f"-(T_out/T0)+{args.beta}*(dp/dp0)"}[args.obj]
    print("\n=== Parametric inverse design result ===")
    print(f"  objective: {obj_desc}")
    print(f"  {'':14s}{'T_out':>10s}{'dp':>10s}{'PEC':>8s}")
    print(f"  {'baseline FEM':14s}{fem_base_q[0]:10.4f}{fem_base_q[1]:10.4f}{1.000:8.3f}")
    print(f"  {'optimised SUR':14s}{T_sur:10.4f}{dp_sur:10.4f}{pec(T_sur,dp_sur):8.3f}")
    print(f"  {'optimised FEM':14s}{fem_opt[0]:10.4f}{fem_opt[1]:10.4f}{pec(*fem_opt):8.3f}")
    if fem_base_q[0] == fem_base_q[0]:
        dT = (fem_opt[0] / fem_base_q[0] - 1) * 100
        dP = (fem_opt[1] / fem_base_q[1] - 1) * 100
        print(f"  FEM-verified vs baseline:  T_out {dT:+.1f}%   dp {dP:+.1f}%   "
              f"PEC {(pec(*fem_opt)-1)*100:+.1f}%")
    print(f"  surrogate vs FEM on optimum:  T_out err "
          f"{abs(T_sur-fem_opt[0])/abs(fem_opt[0])*100:.1f}%  "
          f"dp err {abs(dp_sur-fem_opt[1])/abs(fem_opt[1])*100:.1f}%")

    _plot(hist, yl_np, yu_np, fem, fem_base, args, T0, dp0)
    print(f"\nfigure -> {args.out}")


def _plot(hist, yl, yu, fem, fem_base, args, T0, dp0):
    fig = plt.figure(figsize=(13, 8))
    xs = np.linspace(0, DOMAIN.L, GRID.gx)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(hist["J"]); ax1.set_title("objective J"); ax1.set_xlabel("iteration"); ax1.grid(alpha=.3)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(np.array(hist["T_out"]) / T0, label="T_out / base")
    ax2.plot(np.array(hist["dp"]) / dp0, label="dp / base")
    ax2.axhline(1, color="k", lw=.6); ax2.legend(); ax2.set_title("QoIs (normalised)"); ax2.grid(alpha=.3)

    ax3 = fig.add_subplot(2, 3, 3)
    xb = np.linspace(0, DOMAIN.L, MESH.nx)
    ax3.fill_between(xs, yl, yu, color="#8fbce6", alpha=.6, label="optimised")
    ax3.plot([0, DOMAIN.L], [0, 0], "k--", lw=.6)
    ax3.plot([0, DOMAIN.L], [H, H], "k--", lw=.6, label="baseline walls")
    ax3.set_title("optimised channel shape"); ax3.legend(fontsize=8); ax3.set_ylim(DOMAIN.ybox_min, DOMAIN.ybox_max)

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
    fig.suptitle(f"Parametric inverse design  |  Re={args.Re:.0f} Pr={args.Pr:.1f} "
                 f"Umean={args.Umean:.1f}  |  obj: {obj_txt}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(args.out, dpi=100); plt.close()


if __name__ == "__main__":
    main()
