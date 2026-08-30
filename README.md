# Thermal-Flow Surrogate for 2D Channels (Navier–Stokes constrained)

A deep-learning **surrogate (proxy) model** that predicts the coupled
**flow + thermal field** `(u_x, u_y, p, T)` inside a fixed-thickness 2D flow
channel of **arbitrary shape**, given the geometry and operating conditions
(Reynolds/Prandtl number, inlet flow rate, pressure drop). It is trained
against a **Navier–Stokes finite-element (FEM) reference solver** and
benchmarked against it.

```
 shape (top/bottom walls)  ┐
 Re, Pr                    ├──►  U-Net surrogate  ──►  u_x, u_y, p, T   (≈ ms)
 flow rate (Umean)         │              ▲
 pressure drop (Δp)        ┘              └── trained on / compared to FEM (≈ s)
```

## Physics

Non-dimensional steady incompressible Navier–Stokes + energy transport
(density ρ = 1):

```
(u·∇)u = −∇p + ν ∇²u      ν = Umean·H_in / Re
∇·u   = 0
(u·∇)T = α ∇²T             α = ν / Pr
```

Boundary conditions: parabolic inlet velocity (prescribed mean → flow rate),
no-slip heated walls (`T = 1`), cold inlet (`T = 0`), do-nothing outlet.

## Pipeline

| Stage | File | What it does |
|-------|------|--------------|
| Geometry | `src/geometry.py` | Random smooth channels (contractions, bumps, wavy walls) + body-fitted triangular mesh |
| FEM reference | `src/fem_solver.py` | Taylor–Hood P2/P1 Navier–Stokes (Picard) + P2 advection–diffusion (streamline-diffusion stabilised), via `scikit-fem` |
| Rasterize | `src/rasterize.py` | Interpolate FEM solution onto a fixed `96×192` Cartesian grid + fluid mask |
| Dataset | `src/generate_dataset.py` | Parallel generation of many (geometry, operating point) → field samples |
| Surrogate | `src/model.py` | U-Net (image-to-image), 8 input channels → 4 field channels |
| Physics loss | `src/physics.py` | FD continuity + momentum residuals → the "N-S constrained" term |
| Train | `src/train.py` | Masked data loss + ramped physics loss (AdamW, cosine, MPS) |
| Evaluate | `src/evaluate.py` | Relative-L2 error vs FEM, Δp regression, field/error plots, speed benchmark |
| Inference | `src/predict.py` | Predict on any user-defined channel shape + compare to a fresh FEM solve |
| Design helpers | `src/design_common.py` | Differentiable geometry→surrogate pipeline + physical objectives (T_out, Δp) |
| Inverse design (parametric) | `src/inverse_design.py` | Optimise a few Fourier wall coefficients by autodiff through the surrogate |
| Inverse design (topology-style) | `src/topopt.py` | Free-form high-DOF wall fields with density filter + TV regulariser |
| Tests | `tests/test_pipeline.py` | Unit + integration tests (`pytest`) |

## Run

```bash
source .venv/bin/activate
# --- forward surrogate (operating-condition model, dp is an input) ---
python -m src.generate_dataset --n 500 --workers 9 --out data/dataset.npz
python -m src.train    --data data/dataset.npz --epochs 120
python -m src.evaluate --data data/dataset.npz --model results/model.pt
python -m src.demo                     # arbitrary user-defined channel demo

# --- inverse design (needs the forward operator: dp is an OUTPUT) ---
python -m src.train --data data/dataset.npz --epochs 120 --no_dp --out results/model_fwd.pt
python -m src.inverse_design --beta 0.4 --steps 200     # parametric
python -m src.topopt         --beta 0.4 --steps 300     # topology-style
pytest tests/ -q
```

## Forward vs. inverse

* **Forward surrogate** (`model.pt`): geometry + `Re, Pr, Umean, Δp` → fields.
  Replaces the FEM *simulation* (≈320× faster). `Δp` is a given input.
* **Forward operator** (`model_fwd.pt`, `--no_dp`): geometry + `Re, Pr, Umean` →
  fields, with `Δp` emerging from the predicted pressure field. This is the
  differentiable map used for **inverse design**.
* **Inverse design**: fix the operating point, **optimise the geometry** to
  maximise heat pick-up `T_out` while penalising pumping cost `Δp`, by
  back-propagating the objective through `model_fwd.pt` (autodiff = adjoint).
  The optimised shape is re-simulated with FEM to verify the gain.
  * *parametric* — few Fourier coefficients (smooth, robust);
  * *topology-style* — free-form wall fields (`2·gx` DOF) with a density
    filter and TV regulariser. Stays a two-wall channel so FEM can verify it;
    genuine interior-obstacle topology change would need obstacle-inclusive
    training data + an unstructured re-mesher (documented next step).

## Results (500-sample dataset, 50 held-out test geometries)

Surrogate vs FEM, mean relative-L2 error over the fluid region:

| field | mean | median | p90 |
|-------|------|--------|-----|
| velocity \|u\| | **4.8 %** | 4.2 % | 5.9 % |
| u_x | 4.8 % | 4.2 % | 6.0 % |
| u_y * | 23 % | 20 % | 37 % |
| pressure p | 15.8 % | 14.8 % | 23 % |
| temperature T | **8.4 %** | 7.2 % | 10 % |

\* u_y is near-zero everywhere, so its *relative* error is inflated by a tiny
denominator; its absolute error is small.

- **Pressure-drop Δp regression:** R² = **0.995**
- **Speed:** FEM ≈ **7.3 s/case**, surrogate ≈ **23 ms/case** → **~320× faster**
  (`python -m src.benchmark_speed`)
- Held-out custom venturi channel (not from the generator): |u| 4.8 %, p 7.4 %, T 7.8 %

Figures in `results/`: `compare_{best,median,worst}.png`, `error_summary.png`,
`dp_scatter.png`, `demo_custom_channel.png`.

The trained checkpoint is `results/model_eval.pt` (= best-val `results/model.pt`).

## Inputs / outputs

**Surrogate input (8 channels)**: fluid mask, x-coord, y-coord, wall-distance,
and 4 broadcast scalars — `Re`, `Pr`, `Umean` (flow), `Δp` (pressure).
**Output (4 channels)**: `u_x, u_y, p, T` on the fluid region.
