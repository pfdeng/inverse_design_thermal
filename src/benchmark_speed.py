"""Clean speed benchmark: FEM solve vs surrogate inference (per case)."""
import time
import numpy as np
import torch

from .geometry import random_walls, build_mesh
from .fem_solver import solve_case
from .predict import load_model, build_input


def main():
    model, stats, device = load_model("results/model_eval.pt")

    # FEM timing: several fresh solves
    fem_times = []
    rng = np.random.default_rng(2024)
    for _ in range(4):
        x, lo, up = random_walls(rng)
        mesh, meta = build_mesh(x, lo, up)
        t0 = time.time()
        sol = solve_case(mesh, meta, rng.uniform(50, 200), rng.uniform(1, 5), 1.0)
        dt = time.time() - t0
        if sol is not None:
            fem_times.append(dt)
    fem_ms = np.median(fem_times) * 1000

    # surrogate timing (warm)
    x, lo, up = random_walls(rng)
    inp, _ = build_input(x, lo, up, 120, 3.0, 1.0, 1.0, stats)
    inp = inp.to(device)
    for _ in range(5):
        _ = model(inp)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    reps = 50
    for _ in range(reps):
        _ = model(inp)
    if device == "mps":
        torch.mps.synchronize()
    sur_ms = (time.time() - t0) / reps * 1000

    print(f"FEM solve (median of {len(fem_times)}): {fem_ms:8.1f} ms/case")
    print(f"surrogate inference (mean of {reps}): {sur_ms:8.2f} ms/case  [{device}]")
    print(f"speed-up: ~{fem_ms/sur_ms:.0f}x")


if __name__ == "__main__":
    main()
