"""Shared configuration for the thermal-flow surrogate project.

All quantities are non-dimensional.  We fix the density rho = 1 and the
inlet mean velocity scale, and vary the Reynolds / Peclet numbers through
the kinematic viscosity / thermal diffusivity.  The reference solver
(scikit-fem) treats these as the governing Navier-Stokes parameters.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DomainConfig:
    L: float = 3.0          # channel length (x)
    H: float = 1.0          # nominal channel height (y), reference gap
    y0: float = 0.0         # nominal bottom wall baseline
    # bounding box used for the fixed rasterization grid
    ybox_min: float = -0.6
    ybox_max: float = 1.6


@dataclass
class MeshConfig:
    nx: int = 96            # logical nodes along x (body-fitted mesh)
    ny: int = 40            # logical nodes across the channel


@dataclass
class GridConfig:
    """Fixed Cartesian grid the surrogate operates on (image representation)."""
    gx: int = 192          # pixels along x
    gy: int = 96           # pixels along y


@dataclass
class PhysicsRanges:
    Re_min: float = 10.0
    Re_max: float = 250.0
    Pr_min: float = 0.7
    Pr_max: float = 7.0
    Umean_min: float = 0.6   # inlet mean velocity (=> flow-rate variation)
    Umean_max: float = 1.4
    T_in: float = 0.0        # cold inlet
    T_wall: float = 1.0      # heated walls


DOMAIN = DomainConfig()
MESH = MeshConfig()
GRID = GridConfig()
PHYS = PhysicsRanges()

SEED = 1234
