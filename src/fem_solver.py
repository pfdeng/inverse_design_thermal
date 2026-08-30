"""Steady incompressible Navier-Stokes + heat FEM reference solver (scikit-fem).

Governing equations (non-dimensional, rho = 1):
    (u . grad) u = -grad p + nu * lap(u)          (momentum)
    div u = 0                                       (continuity)
    (u . grad) T = alpha * lap(T)                   (energy / advection-diffusion)

Discretisation:
    * Taylor-Hood elements: P2 velocity (vector) + P1 pressure.
    * Momentum-continuity solved as a monolithic saddle-point system with
      Picard (fixed-point) linearisation of the convective term.
    * Temperature: P2, Galerkin + streamline-diffusion (SUPG-like) stabilisation.

Boundary conditions:
    inlet  (x=0):   parabolic u with prescribed mean velocity, T = T_in (cold)
    walls:          no-slip u = 0, T = T_wall (heated)
    outlet (x=L):   do-nothing (natural) -> fixes the pressure datum, dT/dn = 0
"""
import numpy as np
import scipy.sparse as sp
from skfem import (Basis, ElementVector, ElementTriP2, ElementTriP1,
                   BilinearForm, asm, condense, solve)
from skfem.models.poisson import vector_laplace

from .config import PHYS


# ----------------------------------------------------------------------------
# weak forms
# ----------------------------------------------------------------------------
@BilinearForm
def _divergence(u, p, w):
    # returns integral of  p * div(u)   -> matrix B[p, u]
    return p * (u.grad[0][0] + u.grad[1][1])


@BilinearForm
def _convection(u, v, w):
    # integral of ((wind . grad) u) . v ,  u.grad[i][j] = du_i/dx_j
    wind = w["wind"]
    conv0 = wind[0] * u.grad[0][0] + wind[1] * u.grad[0][1]
    conv1 = wind[0] * u.grad[1][0] + wind[1] * u.grad[1][1]
    return conv0 * v[0] + conv1 * v[1]


def _temperature_form(alpha):
    @BilinearForm
    def adv_diff(T, s, w):
        wind = w["wind"]
        wx, wy = wind[0], wind[1]
        speed = np.sqrt(wx ** 2 + wy ** 2) + 1e-9
        adv = (wx * T.grad[0] + wy * T.grad[1]) * s
        diff = alpha * (T.grad[0] * s.grad[0] + T.grad[1] * s.grad[1])
        # streamline-diffusion stabilisation
        tau = 0.5 * w.h / speed
        wgt = wx * T.grad[0] + wy * T.grad[1]
        wgs = wx * s.grad[0] + wy * s.grad[1]
        return adv + diff + tau * wgt * wgs
    return adv_diff


# ----------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------
def solve_case(mesh, meta, Re, Pr, Umean,
               picard_iters=8, picard_tol=1e-5, relax=0.7, verbose=False):
    """Solve one thermal-flow case on a body-fitted mesh.

    Returns a dict of vertex fields and scalar operating parameters, or None
    if the solve fails to converge / is unstable.
    """
    T_in, T_wall = PHYS.T_in, PHYS.T_wall

    ub = Basis(mesh, ElementVector(ElementTriP2()), intorder=4)
    pb = Basis(mesh, ElementTriP1(), intorder=4)
    tb = Basis(mesh, ElementTriP2(), intorder=4)

    uN, pN = ub.N, pb.N
    nvert = mesh.nvertices

    # ---- geometry-derived quantities -------------------------------------
    x = meta["x"]
    low0, up0 = meta["y_low"][0], meta["y_up"][0]
    H_in = up0 - low0
    nu = Umean * H_in / Re                 # kinematic viscosity from Re
    alpha = nu / Pr                        # thermal diffusivity from Pr
    Q = Umean * H_in                       # inlet volumetric flow rate (per unit depth)

    # ---- boundary facets --------------------------------------------------
    inlet = mesh.facets_satisfying(lambda p: np.isclose(p[0], 0.0))
    outlet = mesh.facets_satisfying(lambda p: np.isclose(p[0], meta["L"]))
    bnd = mesh.boundary_facets()
    walls = np.setdiff1d(bnd, np.concatenate([inlet, outlet]))

    inlet_dofs = ub.get_dofs(inlet).all()
    wall_dofs = ub.get_dofs(walls).all()

    # ---- Dirichlet values on velocity ------------------------------------
    u_dir = np.zeros(uN)
    # inlet parabolic profile on the x-component dofs (even indices)
    loc = ub.doflocs
    for d in inlet_dofs:
        if d % 2 == 0:  # x-component
            y = loc[1, d]
            eta = (y - low0) / max(H_in, 1e-9)
            eta = np.clip(eta, 0.0, 1.0)
            u_dir[d] = 6.0 * Umean * eta * (1.0 - eta)   # mean = Umean, u_y=0
        # y-component and all wall dofs stay 0
    dir_vel_dofs = np.unique(np.concatenate([inlet_dofs, wall_dofs]))

    # ---- static blocks ----------------------------------------------------
    A = asm(vector_laplace, ub)            # int grad u : grad v
    B = asm(_divergence, ub, pb)           # B[p,u] = int p div u
    zero_pp = sp.csr_matrix((pN, pN))

    # combined Dirichlet dof set (velocity block occupies [0, uN))
    D = dir_vel_dofs
    x0 = np.zeros(uN + pN)
    x0[:uN] = u_dir

    # ---- Picard iterations -----------------------------------------------
    u_sol = np.zeros(uN)
    u_prev_full = np.zeros(uN + pN)
    u_prev_full[:uN] = u_dir
    for it in range(picard_iters):
        C = asm(_convection, ub, ub, wind=ub.interpolate(u_sol))
        K11 = nu * A + C
        K = sp.bmat([[K11, -B.T],
                     [-B,  zero_pp]], format="csr")
        f = np.zeros(uN + pN)
        try:
            sol = solve(*condense(K, f, x=x0, D=D))
        except Exception as e:  # noqa
            if verbose:
                print("  solve failed:", e)
            return None
        u_new = sol[:uN]
        if not np.all(np.isfinite(u_new)):
            return None
        # under-relaxation
        u_relaxed = relax * u_new + (1 - relax) * u_sol
        denom = np.linalg.norm(u_new) + 1e-12
        change = np.linalg.norm(u_relaxed - u_sol) / denom
        u_sol = u_relaxed
        p_sol = sol[uN:]
        if verbose:
            print(f"  picard {it}: change={change:.2e}")
        if change < picard_tol and it >= 1:
            break

    if np.linalg.norm(u_sol) < 1e-8:
        return None

    # ---- temperature solve ------------------------------------------------
    adv_diff = _temperature_form(alpha)
    KT = asm(adv_diff, tb, tb, wind=ub.interpolate(u_sol))
    fT = np.zeros(tb.N)
    T_dir = np.zeros(tb.N)
    tin_dofs = tb.get_dofs(inlet).all()
    twall_dofs = tb.get_dofs(walls).all()
    T_dir[tin_dofs] = T_in
    T_dir[twall_dofs] = T_wall
    DT = np.unique(np.concatenate([tin_dofs, twall_dofs]))
    xT = np.zeros(tb.N)
    xT[DT] = T_dir[DT]
    try:
        T_sol = solve(*condense(KT, fT, x=xT, D=DT))
    except Exception:
        return None
    if not np.all(np.isfinite(T_sol)):
        return None

    # ---- extract vertex fields -------------------------------------------
    ux = u_sol[0:2 * nvert:2]
    uy = u_sol[1:2 * nvert:2]
    p = p_sol[:nvert]
    T = T_sol[:nvert]

    # pressure drop (inlet mean - outlet mean), outlet datum ~ 0
    inlet_vtx = np.where(np.isclose(mesh.p[0], 0.0))[0]
    outlet_vtx = np.where(np.isclose(mesh.p[0], meta["L"]))[0]
    dp = float(p[inlet_vtx].mean() - p[outlet_vtx].mean())

    # basic sanity: bounded temperature
    if T.min() < -0.2 or T.max() > 1.2:
        # mild overshoot tolerated; large -> reject
        if T.min() < -0.5 or T.max() > 1.5:
            return None

    return {
        "coords": mesh.p.copy(),          # (2, nvert)
        "tris": mesh.t.copy(),            # (3, ntri)
        "ux": ux, "uy": uy, "p": p, "T": T,
        "Re": float(Re), "Pr": float(Pr), "Umean": float(Umean),
        "nu": float(nu), "alpha": float(alpha),
        "Q": float(Q), "dp": dp, "H_in": float(H_in),
    }
