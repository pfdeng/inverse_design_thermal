"""Evaluate the surrogate against the FEM reference: metrics, plots, speed."""
import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import load_dataset
from .model import UNet
from .physics import ns_residuals, erode_mask
from .config import DOMAIN
from .geometry import random_walls, build_mesh
from .fem_solver import solve_case
from .rasterize import rasterize

FIELD_NAMES = ["u_x", "u_y", "p", "T"]
EXTENT = [0, DOMAIN.L, DOMAIN.ybox_min, DOMAIN.ybox_max]


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    base = ckpt["args"]["base"]
    in_ch = ckpt["stats"].get("in_ch", 8)
    model = UNet(in_ch=in_ch, out_ch=4, base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["stats"]


def predict(model, inp, stats, device):
    fmean = torch.tensor(stats["fmean"], device=device)
    fstd = torch.tensor(stats["fstd"], device=device)
    with torch.no_grad():
        pn = model(inp.to(device))
    phys = pn * fstd.view(1, -1, 1, 1) + fmean.view(1, -1, 1, 1)
    return phys.cpu().numpy()


def rel_l2(pred, true, mask):
    m = mask > 0.5
    num = np.sqrt(((pred - true) ** 2 * m).sum())
    den = np.sqrt((true ** 2 * m).sum()) + 1e-12
    return num / den


def dp_from_field(p, mask):
    """pressure drop = mean inlet-column p - mean outlet-column p (fluid only)."""
    m = mask > 0.5
    cols_in, cols_out = [], []
    for c in range(3):
        col = p[:, c][m[:, c]]
        if col.size:
            cols_in.append(col.mean())
    for c in range(-3, 0):
        col = p[:, c][m[:, c]]
        if col.size:
            cols_out.append(col.mean())
    return float(np.mean(cols_in) - np.mean(cols_out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--model", default="results/model.pt")
    ap.add_argument("--n_show", type=int, default=3)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, stats = load_model(args.model, device)
    data = load_dataset(args.data, use_dp=stats.get("use_dp", True))
    inp, out, mask, nu = data["test"]
    te_idx = data["test_idx"]
    raw = data["raw"]

    pred = predict(model, inp, stats, device)      # (Nte,4,gy,gx) physical
    true = raw["fields"][te_idx]                    # (Nte,4,gy,gx) FEM
    m = raw["mask"][te_idx]                         # (Nte,gy,gx)
    scal = raw["scalars"][te_idx]

    # ---- per-field relative L2 errors ------------------------------------
    Nte = pred.shape[0]
    errs = {name: [] for name in FIELD_NAMES}
    errs["|u|"] = []
    dp_true, dp_pred = [], []
    for i in range(Nte):
        for k, name in enumerate(FIELD_NAMES):
            errs[name].append(rel_l2(pred[i, k], true[i, k], m[i]))
        sp_p = np.sqrt(pred[i, 0] ** 2 + pred[i, 1] ** 2)
        sp_t = np.sqrt(true[i, 0] ** 2 + true[i, 1] ** 2)
        errs["|u|"].append(rel_l2(sp_p, sp_t, m[i]))
        dp_true.append(dp_from_field(true[i, 2], m[i]))
        dp_pred.append(dp_from_field(pred[i, 2], m[i]))

    print("\n=== Surrogate vs FEM: mean relative L2 error over test set ===")
    print(f"  test cases: {Nte}")
    for name in ["|u|", "u_x", "u_y", "p", "T"]:
        e = np.array(errs[name])
        print(f"  {name:>4s}: mean {e.mean()*100:5.2f}%  median {np.median(e)*100:5.2f}%  "
              f"p90 {np.percentile(e,90)*100:5.2f}%")

    dp_true, dp_pred = np.array(dp_true), np.array(dp_pred)
    dp_r2 = 1 - np.sum((dp_pred - dp_true) ** 2) / (np.sum((dp_true - dp_true.mean()) ** 2) + 1e-12)
    print(f"  pressure-drop dp: R^2 = {dp_r2:.3f}")

    # ---- physics residual check (continuity) -----------------------------
    pt = torch.tensor(pred)
    tt = torch.tensor(true)
    mt = torch.tensor(m)
    nut = torch.tensor(scal[:, 2] * scal[:, 5] / scal[:, 0])  # Umean*H_in/Re
    em = erode_mask(mt, 3)
    cP, _, _ = ns_residuals(pt, nut, mt)
    cF, _, _ = ns_residuals(tt, nut, mt)
    div_pred = (cP.abs() * em).sum() / (em.sum() + 1e-6)
    div_fem = (cF.abs() * em).sum() / (em.sum() + 1e-6)
    print(f"  mean |div u|: surrogate {div_pred:.4f} | FEM {div_fem:.4f}")

    # ---- speed benchmark --------------------------------------------------
    # surrogate throughput
    x1 = inp[:1].to(device)
    for _ in range(3):
        _ = model(x1)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    reps = 20
    for _ in range(reps):
        _ = model(x1)
    if device == "mps":
        torch.mps.synchronize()
    t_sur = (time.time() - t0) / reps
    # FEM cost (one fresh solve)
    rng = np.random.default_rng(777)
    xx, lo, up = random_walls(rng)
    mesh, meta = build_mesh(xx, lo, up)
    t0 = time.time(); _ = solve_case(mesh, meta, 120, 3.0, 1.0); t_fem = time.time() - t0
    print(f"\n=== Speed (per case) ===")
    print(f"  FEM solve : {t_fem*1000:8.1f} ms")
    print(f"  surrogate : {t_sur*1000:8.1f} ms   -> speed-up ~{t_fem/t_sur:.0f}x")

    # ---- figures: worst/median/best cases --------------------------------
    order = np.argsort(errs["T"])
    picks = [order[0], order[len(order)//2], order[-1]]   # best, median, worst by T
    labels = ["best", "median", "worst"]
    for pick, lab in zip(picks, labels):
        _plot_case(pred[pick], true[pick], m[pick], scal[pick], errs, pick,
                   f"results/compare_{lab}.png", lab)
    _plot_scatter(dp_true, dp_pred, dp_r2, "results/dp_scatter.png")
    _plot_error_summary(errs, "results/error_summary.png")
    print("\nFigures written to results/: compare_best/median/worst.png, "
          "dp_scatter.png, error_summary.png")


def _plot_case(pred, true, mask, scal, errs, idx, path, lab):
    speed_t = np.sqrt(true[0] ** 2 + true[1] ** 2)
    speed_p = np.sqrt(pred[0] ** 2 + pred[1] ** 2)
    rows = [("|u|", speed_t, speed_p, "turbo"),
            ("p", true[2], pred[2], "coolwarm"),
            ("T", true[3], pred[3], "inferno")]
    fig, ax = plt.subplots(3, 3, figsize=(13, 7.5))
    for r, (name, ft, fp, cmap) in enumerate(rows):
        ftm = np.where(mask > 0.5, ft, np.nan)
        fpm = np.where(mask > 0.5, fp, np.nan)
        err = np.where(mask > 0.5, np.abs(fp - ft), np.nan)
        vmin, vmax = np.nanmin(ftm), np.nanmax(ftm)
        for c, (img, title, cm, vm) in enumerate([
                (ftm, f"FEM  {name}", cmap, (vmin, vmax)),
                (fpm, f"Surrogate  {name}", cmap, (vmin, vmax)),
                (err, f"|error|  {name}", "magma", (None, None))]):
            im = ax[r, c].imshow(img, origin="lower", aspect="auto", extent=EXTENT,
                                 cmap=cm, vmin=vm[0], vmax=vm[1])
            ax[r, c].set_title(title, fontsize=10)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            plt.colorbar(im, ax=ax[r, c], fraction=0.035)
    Re, Pr, Um = scal[0], scal[1], scal[2]
    fig.suptitle(f"[{lab}] test case #{idx}  |  Re={Re:.0f}  Pr={Pr:.1f}  Umean={Um:.2f}  "
                 f"| relL2:  |u| {errs['|u|'][idx]*100:.1f}%  p {errs['p'][idx]*100:.1f}%  "
                 f"T {errs['T'][idx]*100:.1f}%", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=95); plt.close()


def _plot_scatter(dt, dp, r2, path):
    fig, a = plt.subplots(figsize=(5, 5))
    a.scatter(dt, dp, s=28, alpha=0.7, edgecolor="k", linewidth=0.4)
    lim = [min(dt.min(), dp.min()), max(dt.max(), dp.max())]
    a.plot(lim, lim, "r--", lw=1)
    a.set_xlabel("FEM pressure drop  Δp"); a.set_ylabel("Surrogate Δp")
    a.set_title(f"Pressure drop:  R² = {r2:.3f}")
    a.set_aspect("equal"); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def _plot_error_summary(errs, path):
    names = ["|u|", "u_x", "u_y", "p", "T"]
    data = [np.array(errs[n]) * 100 for n in names]
    fig, a = plt.subplots(figsize=(7, 4.2))
    bp = a.boxplot(data, labels=names, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#6db3f2")
    a.set_ylabel("relative L2 error (%)")
    a.set_title("Surrogate vs FEM error distribution (test set)")
    a.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


if __name__ == "__main__":
    main()
