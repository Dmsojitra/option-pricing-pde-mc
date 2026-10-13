"""Monte Carlo pricing: European options with variance reduction, American
options with Longstaff-Schwartz least-squares regression.

Monte Carlo error shrinks like 1/sqrt(n_paths) whatever the dimension, which is
why it is the method of choice for path-dependent and multi-asset products.
Variance reduction lowers the constant in front of that rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self):
        return (self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error)


def _terminal_prices(S0, T, r, sigma, q, z):
    # Exact GBM solution: no time-stepping error for European payoffs.
    return S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)


def price_european_mc(S0, K, T, r, sigma, q=0.0, kind="call", n_paths=100_000,
                      method="plain", seed=None) -> MCResult:
    """European option by Monte Carlo.

    method:
      'plain'      : standard estimator
      'antithetic' : pair each normal draw z with -z (negatively correlated payoffs)
      'control'    : control variate with the discounted terminal stock price,
                     whose expectation S0 e^{-qT} is known exactly
    """
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    sign = 1.0 if kind == "call" else -1.0

    def payoff(ST):
        return np.maximum(sign * (ST - K), 0.0)

    if method == "antithetic":
        z = rng.standard_normal(n_paths // 2)
        y = 0.5 * disc * (payoff(_terminal_prices(S0, T, r, sigma, q, z))
                          + payoff(_terminal_prices(S0, T, r, sigma, q, -z)))
        return MCResult(float(y.mean()), float(y.std(ddof=1) / np.sqrt(len(y))), n_paths)

    z = rng.standard_normal(n_paths)
    ST = _terminal_prices(S0, T, r, sigma, q, z)
    y = disc * payoff(ST)
    if method == "plain":
        return MCResult(float(y.mean()), float(y.std(ddof=1) / np.sqrt(n_paths)), n_paths)
    if method == "control":
        x = disc * ST
        ex = S0 * np.exp(-q * T)
        cov = np.cov(y, x)
        beta = cov[0, 1] / cov[1, 1]
        yc = y - beta * (x - ex)
        return MCResult(float(yc.mean()), float(yc.std(ddof=1) / np.sqrt(n_paths)), n_paths)
    raise ValueError("method must be 'plain', 'antithetic' or 'control'")


def price_american_lsm(S0, K, T, r, sigma, q=0.0, kind="put", n_paths=100_000,
                       n_steps=50, degree=3, seed=None) -> MCResult:
    """American option by Longstaff-Schwartz (2001).

    Walk backwards in time. At each exercise date, regress the discounted future
    cash flow on polynomials of the current price (in-the-money paths only) to
    estimate the continuation value; exercise where the payoff beats it.
    Antithetic paths are used to reduce variance.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2
    z = rng.standard_normal((half, n_steps))
    z = np.vstack([z, -z])
    increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    S = S0 * np.exp(np.cumsum(increments, axis=1))  # shape (paths, steps); column k = time (k+1) dt
    sign = 1.0 if kind == "call" else -1.0

    def payoff(s):
        return np.maximum(sign * (s - K), 0.0)

    cash = payoff(S[:, -1])  # cash flow at maturity
    disc = np.exp(-r * dt)
    for k in range(n_steps - 2, -1, -1):
        cash *= disc
        s = S[:, k]
        ex_val = payoff(s)
        itm = ex_val > 0
        if itm.sum() > degree + 1:
            xs = s[itm] / K  # scale for a well-conditioned regression
            A = np.vander(xs, degree + 1)
            coef, *_ = np.linalg.lstsq(A, cash[itm], rcond=None)
            cont = A @ coef
            exercise = ex_val[itm] > cont
            idx = np.flatnonzero(itm)[exercise]
            cash[idx] = ex_val[idx]
    cash *= disc  # from t = dt back to t = 0
    # Exercise at t = 0 is also allowed
    value = max(float(cash.mean()), float(payoff(np.array([S0]))[0]))
    # Standard error using antithetic pairs
    pair = 0.5 * (cash[:half] + cash[half:])
    return MCResult(value, float(pair.std(ddof=1) / np.sqrt(half)), n_paths)


def price_binomial(S0, K, T, r, sigma, q=0.0, kind="put", american=True, n_steps=2000) -> float:
    """Cox-Ross-Rubinstein binomial tree (vectorized), used as a reference."""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(n_steps + 1)
    S = S0 * u ** (n_steps - j) * d ** j
    sign = 1.0 if kind == "call" else -1.0
    V = np.maximum(sign * (S - K), 0.0)
    for n in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            S = S[:-1] / u
            V = np.maximum(V, sign * (S - K))
    return float(V[0])
