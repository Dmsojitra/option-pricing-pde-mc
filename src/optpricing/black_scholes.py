"""Black-Scholes closed-form prices, Greeks and implied volatility.

Model: under the risk-neutral measure the stock follows
    dS = (r - q) S dt + sigma S dW
and a European option price V(S, t) solves the Black-Scholes PDE
    V_t + 1/2 sigma^2 S^2 V_SS + (r - q) S V_S - r V = 0.

For calls and puts this PDE has the closed-form solution below, which we use as
the ground truth to measure the error of the numerical methods.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def price(S, K, T, r, sigma, q=0.0, kind="call"):
    """European option price. kind = 'call' or 'put'. Works on arrays."""
    S, K, T = np.asarray(S, float), np.asarray(K, float), np.asarray(T, float)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if kind == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def greeks(S, K, T, r, sigma, q=0.0, kind="call") -> dict:
    """Delta, gamma, vega (per 1.00 vol), theta (per year), rho."""
    S, K, T = np.asarray(S, float), np.asarray(K, float), np.asarray(T, float)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf = norm.pdf(d1)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    gamma = disc_q * pdf / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf * np.sqrt(T)
    if kind == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * np.sqrt(T))
                 - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf * sigma / (2 * np.sqrt(T))
                 + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(target_price, S, K, T, r, q=0.0, kind="call",
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """Volatility that reproduces a given price (Brent root finding).

    Returns NaN if the price is outside the no-arbitrage bounds.
    """
    intrinsic = max(0.0, (S * np.exp(-q * T) - K * np.exp(-r * T)) * (1 if kind == "call" else -1))
    if target_price <= intrinsic + 1e-12:
        return np.nan

    def f(sig):
        return price(S, K, T, r, sig, q, kind) - target_price

    if f(lo) * f(hi) > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-10)
