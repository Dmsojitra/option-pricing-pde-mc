"""Finite-difference solver for the Black-Scholes PDE (European and American).

Change of variables x = ln S and tau = T - t (time to maturity) turns the PDE into
one with constant coefficients:

    V_tau = 1/2 sigma^2 V_xx + nu V_x - r V,     nu = r - q - sigma^2 / 2

Discretization
--------------
* Space: uniform grid in x, central differences (second order).
* Time: theta-scheme. theta = 1/2 is Crank-Nicolson (second order in time).
* Rannacher start-up: the payoff has a kink at the strike, which makes plain
  Crank-Nicolson produce spurious oscillations in Delta and Gamma. Taking the first
  few steps fully implicit (theta = 1) damps them and restores second-order
  convergence. This is standard practice in banks.
* American options: at each time step the price must stay above the payoff
  (a linear complementarity problem). We solve it exactly with the
  Brennan-Schwartz algorithm, a tridiagonal solve with a max() built in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded


@dataclass
class FDResult:
    price: float
    delta: float
    gamma: float
    S_grid: np.ndarray
    V_grid: np.ndarray
    exercise_boundary: np.ndarray | None = None  # (tau, S*) for American options


def _payoff(S, K, kind):
    return np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)


def _brennan_schwartz(a, b, c, d, g, exercise_low: bool):
    """Solve tridiagonal A v = d subject to v >= g (American constraint).

    a: sub-diagonal (a[0] unused), b: diagonal, c: super-diagonal (c[-1] unused).
    exercise_low=True when early exercise happens at low prices (puts).
    """
    n = len(b)
    bb = b.astype(float).copy()
    dd = d.astype(float).copy()
    v = np.empty(n)
    if exercise_low:
        # eliminate from the top, substitute from the bottom with max()
        for i in range(n - 2, -1, -1):
            m = c[i] / bb[i + 1]
            bb[i] -= m * a[i + 1]
            dd[i] -= m * dd[i + 1]
        v[0] = max(dd[0] / bb[0], g[0])
        for i in range(1, n):
            v[i] = max((dd[i] - a[i] * v[i - 1]) / bb[i], g[i])
    else:
        for i in range(1, n):
            m = a[i] / bb[i - 1]
            bb[i] -= m * c[i - 1]
            dd[i] -= m * dd[i - 1]
        v[-1] = max(dd[-1] / bb[-1], g[-1])
        for i in range(n - 2, -1, -1):
            v[i] = max((dd[i] - c[i] * v[i + 1]) / bb[i], g[i])
    return v


def price_fd(S0, K, T, r, sigma, q=0.0, kind="call", american=False,
             n_space: int = 400, n_time: int = 400, theta: float = 0.5,
             rannacher_steps: int = 4, width_sd: float = 6.0) -> FDResult:
    """Price a European or American option with a theta-scheme on x = ln S.

    n_space    : number of space intervals
    n_time     : number of time steps
    theta      : 0.5 = Crank-Nicolson, 1.0 = fully implicit
    width_sd   : the grid covers ln S0 +/- width_sd * sigma * sqrt(T)
    """
    # Grid centred so that ln S0 lies exactly on a node
    half = width_sd * sigma * np.sqrt(T)
    x = np.linspace(np.log(S0) - half, np.log(S0) + half, n_space + 1)
    dx = x[1] - x[0]
    S = np.exp(x)
    dt = T / n_time
    nu = r - q - 0.5 * sigma ** 2

    # Operator L V = l V_{i-1} + m V_i + u V_{i+1}
    lo = 0.5 * sigma ** 2 / dx ** 2 - nu / (2 * dx)
    mid = -sigma ** 2 / dx ** 2 - r
    up = 0.5 * sigma ** 2 / dx ** 2 + nu / (2 * dx)

    payoff = _payoff(S, K, kind)
    V = payoff.copy()
    ni = n_space - 1  # interior nodes

    def boundary(tau):
        """Dirichlet values at S_min and S_max."""
        if kind == "call":
            lo_v = 0.0
            hi_v = S[-1] * np.exp(-q * tau) - K * np.exp(-r * tau)
            if american:
                hi_v = max(hi_v, S[-1] - K)
        else:
            lo_v = K * np.exp(-r * tau) - S[0] * np.exp(-q * tau)
            if american:
                lo_v = max(lo_v, K - S[0])
            hi_v = 0.0
        return lo_v, hi_v

    boundary_track = []
    tau = 0.0
    for n in range(n_time):
        th = 1.0 if n < rannacher_steps else theta
        tau_new = tau + dt
        # LHS: (I - th dt L), RHS: (I + (1-th) dt L) V
        a = np.full(ni, -th * dt * lo)
        b = np.full(ni, 1 - th * dt * mid)
        c = np.full(ni, -th * dt * up)
        Vi = V[1:-1]
        rhs = Vi + (1 - th) * dt * (lo * V[:-2] + mid * Vi + up * V[2:])
        # old boundary values enter the explicit part through V[:-2] and V[2:]
        lb_new, ub_new = boundary(tau_new)
        rhs[0] += th * dt * lo * lb_new
        rhs[-1] += th * dt * up * ub_new

        if american:
            Vi_new = _brennan_schwartz(a, b, c, rhs, payoff[1:-1], exercise_low=(kind == "put"))
            # record the early-exercise boundary
            ex = np.flatnonzero(np.isclose(Vi_new, payoff[1:-1]) & (payoff[1:-1] > 0))
            if ex.size:
                s_star = S[1:-1][ex.max()] if kind == "put" else S[1:-1][ex.min()]
                boundary_track.append((tau_new, s_star))
        else:
            ab = np.vstack([np.r_[0, c[:-1]], b, np.r_[a[1:], 0]])
            Vi_new = solve_banded((1, 1), ab, rhs)
        V = np.r_[lb_new, Vi_new, ub_new]
        tau = tau_new

    # Price and Greeks at S0 (node at the centre). Greeks via chain rule from x to S.
    i0 = n_space // 2
    V_x = (V[i0 + 1] - V[i0 - 1]) / (2 * dx)
    V_xx = (V[i0 + 1] - 2 * V[i0] + V[i0 - 1]) / dx ** 2
    delta = V_x / S0
    gamma = (V_xx - V_x) / S0 ** 2
    spline = CubicSpline(S, V)
    return FDResult(float(spline(S0)), float(delta), float(gamma), S, V,
                    np.array(boundary_track) if american else None)
