"""Cosmological distance utilities for Fisher forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

C_KMS = 299792.458


@dataclass
class CosmoParams:
    Om: float = 0.3
    w0: float = -1.0
    wa: float = 0.0
    H0: float = 70.0
    M: float = 0.0


@dataclass
class FisherSetup:
    vary_params: Tuple[str, ...]
    fid: CosmoParams = field(default_factory=CosmoParams)
    step_frac: Dict[str, float] = field(
        default_factory=lambda: {"Om": 1e-3, "w0": 1e-3, "wa": 1e-3, "M": 1e-3}
    )


def Ez_flat_w0wa(z: np.ndarray, p: CosmoParams) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    Om = p.Om
    Ode = 1.0 - Om
    de_factor = (1.0 + z) ** (3.0 * (1.0 + p.w0 + p.wa)) * np.exp(
        -3.0 * p.wa * z / (1.0 + z)
    )
    return np.sqrt(Om * (1.0 + z) ** 3 + Ode * de_factor)


def DC_Mpc(z: np.ndarray, p: CosmoParams, n_steps: int = 4096) -> np.ndarray:
    z = np.atleast_1d(z).astype(float)
    zmax = np.max(z)
    zz = np.linspace(0.0, zmax, n_steps)
    invE = 1.0 / Ez_flat_w0wa(zz, p)
    if len(zz) % 2 == 0:
        zz = zz[:-1]
        invE = invE[:-1]
    primitive = np.cumsum((invE[:-1] + invE[1:]) * 0.5 * (zz[1:] - zz[:-1]))
    primitive = np.concatenate([[0.0], primitive])
    integral = np.interp(z, zz, primitive)
    return (C_KMS / p.H0) * integral


def DL_Mpc(z: np.ndarray, p: CosmoParams) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return (1.0 + z) * DC_Mpc(z, p)


def mu_theory(z: np.ndarray, p: CosmoParams) -> np.ndarray:
    """Distance modulus μ(z;p) for flat w0–wa CDM."""
    dl = DL_Mpc(z, p)
    return 5.0 * np.log10(dl) + 25.0 + p.M


def jacobian_mu(z: np.ndarray, p: CosmoParams, setup: FisherSetup) -> np.ndarray:
    """Return ∂μ/∂θ Jacobian columns for θ in ``setup.vary_params`` + ``'M'``."""
    J_cols = []
    for name in setup.vary_params + ("M",):
        if name == "M":
            J_cols.append(np.ones_like(z, dtype=float))
            continue
        step = setup.step_frac.get(name, 1e-3) * getattr(p, name)
        if step == 0.0:
            step = 1e-5
        p_plus = CosmoParams(**vars(p))
        p_minus = CosmoParams(**vars(p))
        setattr(p_plus, name, getattr(p_plus, name) + step)
        setattr(p_minus, name, getattr(p_minus, name) - step)
        mu_p = mu_theory(z, p_plus)
        mu_m = mu_theory(z, p_minus)
        J_cols.append((mu_p - mu_m) / (2.0 * step))
    J = np.vstack(J_cols).T
    return J
