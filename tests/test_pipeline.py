"""Test suite for the thermal-flow surrogate + inverse-design pipeline.

Fast tests use tiny meshes / synthetic stats so they do not depend on the
trained model.  Tests that need a trained model skip gracefully if absent.
Run:  .venv/bin/python -m pytest tests/ -q
"""
import os
import numpy as np
import torch
import pytest

from src.config import GRID, DOMAIN
from src import geometry, fem_solver, rasterize, physics, model as model_mod
from src import design_common as dc
from src import inverse_design as idz
from src import topopt

FWD_MODEL = "results/model_fwd.pt"
DATASET = "data/dataset.npz"


# --------------------------------------------------------------------------- geometry
def test_random_walls_valid():
    rng = np.random.default_rng(0)
    x, lo, up = geometry.random_walls(rng, nx=48)
    assert x.shape == lo.shape == up.shape == (48,)
    assert np.all(up - lo > 0), "upper wall must be above lower wall"
    assert (up - lo).min() >= 0.27 * DOMAIN.H, "minimum gap must be enforced"
    assert np.all(np.diff(x) > 0), "x must be increasing"


def test_build_mesh_shape():
    rng = np.random.default_rng(1)
    x, lo, up = geometry.random_walls(rng, nx=30)
    mesh, meta = geometry.build_mesh(x, lo, up, ny=12)
    assert mesh.nvertices == 30 * 12
    assert meta["nx"] == 30 and meta["ny"] == 12


# --------------------------------------------------------------------------- FEM
def test_fem_solver_physical():
    rng = np.random.default_rng(2)
    x, lo, up = geometry.random_walls(rng, nx=48)
    mesh, meta = geometry.build_mesh(x, lo, up, ny=18)
    sol = fem_solver.solve_case(mesh, meta, Re=80, Pr=2.0, Umean=1.0)
    assert sol is not None
    # temperature bounded between inlet (0) and wall (1)
    assert sol["T"].min() > -0.2 and sol["T"].max() < 1.2
    # parabolic inlet peak ~ 1.5 * Umean
    assert 1.2 < sol["ux"].max() < 1.9
    # pressure drops from inlet to outlet -> positive dp
    assert sol["dp"] > 0


# --------------------------------------------------------------------------- rasterize
def test_rasterize_shapes():
    rng = np.random.default_rng(3)
    x, lo, up = geometry.random_walls(rng, nx=48)
    mesh, meta = geometry.build_mesh(x, lo, up, ny=18)
    sol = fem_solver.solve_case(mesh, meta, Re=60, Pr=1.5, Umean=1.0)
    g = rasterize.rasterize(sol)
    for k in ["ux", "uy", "p", "T", "mask"]:
        assert g[k].shape == (GRID.gy, GRID.gx)
    frac = g["mask"].mean()
    assert 0.1 < frac < 0.9, "mask fluid fraction should be reasonable"


# --------------------------------------------------------------------------- physics
def test_divergence_free_field():
    # uniform horizontal flow -> divergence ~ 0
    fields = torch.zeros(1, 4, GRID.gy, GRID.gx)
    fields[:, 0] = 1.0                      # ux = const
    nu = torch.tensor([0.01])
    mask = torch.ones(1, GRID.gy, GRID.gx)
    cont, mx, my = physics.ns_residuals(fields, nu, mask)
    assert cont.abs().max() < 1e-5

    # linearly varying ux in x -> known divergence dux/dx = 1
    xx = torch.linspace(0, (GRID.gx - 1) * physics.DX, GRID.gx)
    fields2 = torch.zeros(1, 4, GRID.gy, GRID.gx)
    fields2[0, 0] = xx[None, :]
    cont2, _, _ = physics.ns_residuals(fields2, nu, mask)
    assert abs(cont2[0, GRID.gy // 2, GRID.gx // 2].item() - 1.0) < 1e-3


# --------------------------------------------------------------------------- model / data
def test_unet_forward_shapes():
    for in_ch in (7, 8):
        net = model_mod.UNet(in_ch=in_ch, out_ch=4, base=8)
        y = net(torch.randn(2, in_ch, GRID.gy, GRID.gx))
        assert y.shape == (2, 4, GRID.gy, GRID.gx)


@pytest.mark.skipif(not os.path.exists(DATASET), reason="dataset not present")
def test_load_dataset_channels():
    from src.data import load_dataset
    d8 = load_dataset(DATASET, use_dp=True)
    d7 = load_dataset(DATASET, use_dp=False)
    assert d8["stats"]["in_ch"] == 8 and d8["train"][0].shape[1] == 8
    assert d7["stats"]["in_ch"] == 7 and d7["train"][0].shape[1] == 7


# --------------------------------------------------------------------------- inverse-design differentiable path
def _synthetic_stats(in_ch=7):
    return {"fmean": np.zeros(4, np.float32), "fstd": np.ones(4, np.float32),
            "sc_mean": np.array([100., 3., 1.], np.float32),
            "sc_std": np.array([50., 2., 0.3], np.float32),
            "wd_mean": 5.0, "wd_std": 3.0, "in_ch": in_ch, "use_dp": False}


def test_soft_mask_differentiable():
    dev = "cpu"
    grids = dc.grid_tensors(dev)
    ylow = torch.full((GRID.gx,), 0.2, requires_grad=True)
    yup = torch.full((GRID.gx,), 0.8, requires_grad=True)
    m = dc.soft_mask_from_walls(ylow, yup, grids["YY"])
    assert m.shape == (GRID.gy, GRID.gx)
    assert 0.0 <= float(m.detach().min()) and float(m.detach().max()) <= 1.0
    m.sum().backward()
    assert ylow.grad is not None and yup.grad is not None


def test_make_input_and_qoi_gradient():
    dev = "cpu"
    stats = _synthetic_stats(7)
    grids = dc.grid_tensors(dev)
    net = model_mod.UNet(in_ch=7, out_ch=4, base=8)
    a = torch.zeros(4, requires_grad=True)
    b = torch.zeros(4, requires_grad=True)
    s = grids["xs"] / DOMAIN.L
    yl, yu = idz.walls_from_params(a, b, s)
    m = dc.soft_mask_from_walls(yl, yu, grids["YY"])
    inp = dc.make_input(m, 150., 4., 1., stats, grids, dev)
    assert inp.shape == (1, 7, GRID.gy, GRID.gx)
    fields = dc.forward_fields(net, inp, stats, dev)
    T_out = dc.outlet_bulk_temperature(fields, m)
    dp = dc.pressure_drop(fields, m)
    assert np.isfinite(float(T_out.detach())) and np.isfinite(float(dp.detach()))
    # gradient of a QoI must reach the design variables (autodiff "adjoint")
    T_out.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert a.grad.abs().sum() > 0


# --------------------------------------------------------------------------- topopt building blocks
def test_density_filter_and_walls():
    dev = "cpu"
    k = topopt.gaussian_kernel(4.0, dev)
    assert abs(float(k.sum()) - 1.0) < 1e-5
    field = torch.randn(GRID.gx)
    filt = topopt.density_filter(field, k)
    assert filt.shape == (GRID.gx,)
    # filtered field is smoother (smaller total variation)
    assert topopt.tv(filt) < topopt.tv(field)

    grids = dc.grid_tensors(dev)
    s = grids["xs"] / DOMAIN.L
    ra = torch.zeros(GRID.gx, requires_grad=True)
    rb = torch.zeros(GRID.gx, requires_grad=True)
    yl, yu = topopt.walls_from_fields(ra, rb, s, k)
    assert torch.all(yu - yl > 0)
    (yu.sum() + yl.sum()).backward()
    assert ra.grad is not None and rb.grad is not None


# --------------------------------------------------------------------------- end-to-end (needs trained model)
@pytest.mark.skipif(not os.path.exists(FWD_MODEL), reason="forward model not trained yet")
def test_parametric_optimize_reduces_objective():
    from src.predict import load_model
    m, stats, dev = load_model(FWD_MODEL)
    grids = dc.grid_tensors(dev)
    _, _, _, hist = idz.optimize(m, stats, dev, grids, Re=150, Pr=4.0,
                                 Umean=1.0, beta=0.4, steps=25)
    # optimisation should make progress: best objective below the initial one
    assert min(hist["J"]) < hist["J"][0], "objective should improve"
