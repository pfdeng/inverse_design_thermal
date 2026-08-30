"""Generate a dataset of thermal-flow cases with the FEM reference solver.

Each sample: a random channel geometry + random operating point (Re, Pr,
Umean), solved with the Navier-Stokes + heat FEM solver and rasterised onto
the fixed Cartesian grid.  Runs in parallel across CPU cores.

Usage:
    python -m src.generate_dataset --n 450 --workers 8 --out data/dataset.npz
"""
import argparse
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import PHYS, GRID, SEED
from .geometry import random_walls, build_mesh
from .fem_solver import solve_case
from .rasterize import rasterize, grid_coords


def _one_sample(idx):
    """Try to produce one valid sample; return dict or None."""
    rng = np.random.default_rng(SEED + 100000 + idx)
    for _attempt in range(6):
        Re = rng.uniform(PHYS.Re_min, PHYS.Re_max)
        Pr = rng.uniform(PHYS.Pr_min, PHYS.Pr_max)
        Umean = rng.uniform(PHYS.Umean_min, PHYS.Umean_max)
        x, lo, up = random_walls(rng)
        mesh, meta = build_mesh(x, lo, up)
        sol = solve_case(mesh, meta, Re, Pr, Umean)
        if sol is None:
            continue
        g = rasterize(sol)
        fields = np.stack([g["ux"], g["uy"], g["p"], g["T"]], axis=0)  # (4,gy,gx)
        if not np.all(np.isfinite(fields)):
            continue
        scalars = np.array([sol["Re"], sol["Pr"], sol["Umean"],
                            sol["Q"], sol["dp"], sol["H_in"]], dtype=np.float32)
        return {"mask": g["mask"].astype(np.float32),
                "fields": fields.astype(np.float32),
                "scalars": scalars}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, default="data/dataset.npz")
    args = ap.parse_args()

    t0 = time.time()
    masks, fields, scalars = [], [], []
    got, tried = 0, 0
    # oversample indices to account for occasional failures
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_one_sample, i): i for i in range(int(args.n * 1.3))}
        for fut in as_completed(futures):
            tried += 1
            res = fut.result()
            if res is not None:
                masks.append(res["mask"])
                fields.append(res["fields"])
                scalars.append(res["scalars"])
                got += 1
                if got % 25 == 0:
                    print(f"  {got} samples ({time.time()-t0:.0f}s)")
            if got >= args.n:
                break
        for fut in futures:
            fut.cancel()

    masks = np.stack(masks)          # (N,gy,gx)
    fields = np.stack(fields)        # (N,4,gy,gx)
    scalars = np.stack(scalars)      # (N,6)
    xs, ys, XX, YY = grid_coords()
    np.savez_compressed(
        args.out,
        mask=masks, fields=fields, scalars=scalars,
        X=XX.astype(np.float32), Y=YY.astype(np.float32),
        scalar_names=np.array(["Re", "Pr", "Umean", "Q", "dp", "H_in"]),
        field_names=np.array(["ux", "uy", "p", "T"]),
    )
    print(f"Saved {got} samples to {args.out} in {time.time()-t0:.0f}s "
          f"(tried {tried}). grid={GRID.gy}x{GRID.gx}")


if __name__ == "__main__":
    main()
