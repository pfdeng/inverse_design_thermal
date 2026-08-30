"""Demo: define an arbitrary 2D channel shape, predict its thermal-flow field
with the surrogate, and compare against a fresh FEM solve side-by-side.

Run:  python -m src.demo
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import DOMAIN
from .predict import load_model, predict, fem_reference

EXTENT = [0, DOMAIN.L, DOMAIN.ybox_min, DOMAIN.ybox_max]


def custom_channel(nx=96):
    """A converging–diverging (venturi) channel with a wavy top wall —
    a shape NOT drawn from the training generator."""
    L, H = DOMAIN.L, DOMAIN.H
    x = np.linspace(0, L, nx)
    s = x / L
    win = np.sin(np.pi * s) ** 2
    gap = H * (1.0 - 0.28 * win)                     # moderate central throat
    center = 0.5 * H + 0.06 * np.sin(2 * np.pi * s) * win
    y_low = center - 0.5 * gap
    y_up = center + 0.5 * gap + 0.03 * np.sin(3 * np.pi * s) * win
    return x, y_low, y_up


def main():
    Re, Pr, Umean = 150.0, 4.0, 1.2
    x, lo, up = custom_channel()

    model, stats, device = load_model("results/model_eval.pt")

    # FEM first (gives us the true Δp to feed the surrogate as the "pressure" input)
    t0 = time.time()
    fem = fem_reference(x, lo, up, Re, Pr, Umean)
    t_fem = time.time() - t0
    if fem is None:
        print("FEM failed on this geometry"); return
    dp = fem["dp"]

    t0 = time.time()
    sur = predict(x, lo, up, Re, Pr, Umean, dp, model, stats, device)
    t_sur = time.time() - t0

    # errors
    m = fem["mask"] > 0.5
    def rl2(a, b):
        return np.sqrt(((a - b) ** 2 * m).sum()) / (np.sqrt((b ** 2 * m).sum()) + 1e-12)
    sp_t = np.sqrt(fem["ux"] ** 2 + fem["uy"] ** 2)
    sp_s = np.sqrt(sur["ux"] ** 2 + sur["uy"] ** 2)
    print(f"Custom venturi channel | Re={Re} Pr={Pr} Umean={Umean} Δp={dp:.3f}")
    print(f"  rel-L2  |u| {rl2(sp_s,sp_t)*100:.1f}%  p {rl2(sur['p'],fem['p'])*100:.1f}%  "
          f"T {rl2(sur['T'],fem['T'])*100:.1f}%")
    print(f"  time: FEM {t_fem*1000:.0f} ms   surrogate {t_sur*1000:.1f} ms   "
          f"(~{t_fem/t_sur:.0f}x faster)")

    rows = [("|u|", sp_t, sp_s, "turbo"),
            ("p", fem["p"], sur["p"], "coolwarm"),
            ("T", fem["T"], sur["T"], "inferno")]
    fig, ax = plt.subplots(3, 3, figsize=(13, 7.5))
    for r, (name, ft, fp, cmap) in enumerate(rows):
        ftm = np.where(m, ft, np.nan); fpm = np.where(m, fp, np.nan)
        err = np.where(m, np.abs(fp - ft), np.nan)
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
    fig.suptitle("Arbitrary user-defined channel: surrogate vs FEM", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("results/demo_custom_channel.png", dpi=100); plt.close()
    print("  figure -> results/demo_custom_channel.png")


if __name__ == "__main__":
    main()
