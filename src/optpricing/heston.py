"""Heston stochastic-volatility model: pricing and calibration.

Black-Scholes assumes one constant volatility, but real option markets show a
volatility *smile/skew*: the implied volatility depends on the strike. Heston lets
the variance v_t follow its own mean-reverting process:

    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,     corr(dW1, dW2) = rho

Parameters: v0 (current variance), kappa (mean-reversion speed), theta (long-run
variance), xi (vol of vol), rho (spot/vol correlation; rho < 0 creates the skew).

European prices come from the characteristic function by Fourier inversion
(semi-closed form), which makes calibration fast. We use the "little Heston trap"
formulation (Albrecher et al., 2007), which is numerically stable for long maturities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from . import black_scholes as bs

# Gauss-Legendre nodes on [0, U_MAX] for the Fourier integral (vectorized, fast)
_U_MAX = 200.0
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(256)
_U = 0.5 * _U_MAX * (_NODES + 1)
_W = 0.5 * _U_MAX * _WEIGHTS
_U[_U == 0] = 1e-10


@dataclass
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    NAMES = ("v0", "kappa", "theta", "xi", "rho")

    def to_vector(self):
        return np.array([self.v0, self.kappa, self.theta, self.xi, self.rho])

    def feller(self) -> bool:
        """Feller condition 2 kappa theta > xi^2: variance never hits zero."""
        return 2 * self.kappa * self.theta > self.xi ** 2


def char_func(u, S0, T, r, q, p: HestonParams):
    """Characteristic function of ln S_T, E[exp(i u ln S_T)]."""
    iu = 1j * u
    b = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(b ** 2 + p.xi ** 2 * (iu + u ** 2))
    g = (b - d) / (b + d)
    edT = np.exp(-d * T)
    C = (p.kappa * p.theta / p.xi ** 2) * ((b - d) * T - 2 * np.log((1 - g * edT) / (1 - g)))
    D = ((b - d) / p.xi ** 2) * (1 - edT) / (1 - g * edT)
    return np.exp(iu * (np.log(S0) + (r - q) * T) + C + D * p.v0)


def price(S0, K, T, r, q, p: HestonParams, kind="call"):
    """European option price(s). K may be an array (vectorized over strikes)."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    lnK = np.log(K)[:, None]
    u = _U[None, :]
    phi = char_func(u, S0, T, r, q, p)
    phi_shift = char_func(u - 1j, S0, T, r, q, p)
    fwd = S0 * np.exp((r - q) * T)  # = phi(-i)
    e = np.exp(-1j * u * lnK)
    P1 = 0.5 + (1 / np.pi) * np.sum(_W * np.real(e * phi_shift / (1j * u * fwd)), axis=1)
    P2 = 0.5 + (1 / np.pi) * np.sum(_W * np.real(e * phi / (1j * u)), axis=1)
    call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    if kind == "call":
        out = call
    else:  # put-call parity
        out = call - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    return out if out.size > 1 else float(out[0])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
LOWER = np.array([1e-4, 0.05, 1e-4, 0.01, -0.99])
UPPER = np.array([1.0, 10.0, 1.0, 2.0, 0.99])


@dataclass
class CalibrationResult:
    params: HestonParams
    rmse_iv_bp: float          # implied-vol RMSE in basis points (0.01 %)
    n_evals: int
    success: bool


def calibrate(quotes, S0, r, q=0.0, x0=(0.04, 1.5, 0.04, 0.5, -0.5)) -> CalibrationResult:
    """Fit Heston parameters to market option prices.

    quotes : DataFrame with columns K, T, price, kind ('call'/'put'), and
             optionally weight. Errors are measured in price divided by the BS vega,
             which approximates implied-vol errors without inverting
             Black-Scholes inside the loop.
    """
    quotes = quotes.reset_index(drop=True)
    groups = [(T, g) for T, g in quotes.groupby("T")]
    mkt_iv = np.array([bs.implied_vol(row.price, S0, row.K, row.T, r, q, row.kind)
                       for row in quotes.itertuples()])
    vega = np.asarray(bs.greeks(S0, quotes.K.values, quotes["T"].values, r,
                                np.nan_to_num(mkt_iv, nan=0.2), q)["vega"])
    vega = np.maximum(vega, 1e-3 * S0)
    w = quotes["weight"].values if "weight" in quotes else np.ones(len(quotes))

    def residuals(x):
        p = HestonParams(*x)
        model = np.empty(len(quotes))
        for T, g in groups:
            for kind in ("call", "put"):
                m = g["kind"] == kind
                if m.any():
                    idx = g.index[m]
                    model[idx] = price(S0, g.K[m].values, T, r, q, p, kind)
        return w * (model - quotes.price.values) / vega

    res = least_squares(residuals, np.asarray(x0), bounds=(LOWER, UPPER), method="trf",
                        x_scale=np.array([0.01, 1.0, 0.01, 0.1, 0.1]))
    p = HestonParams(*res.x)
    model_iv = model_implied_vols(quotes, S0, r, q, p)
    rmse = float(np.sqrt(np.nanmean((model_iv - mkt_iv) ** 2)) * 1e4)
    return CalibrationResult(p, rmse, int(res.nfev), bool(res.success))


def model_implied_vols(quotes, S0, r, q, p: HestonParams) -> np.ndarray:
    out = []
    for row in quotes.itertuples():
        mp = price(S0, row.K, row.T, r, q, p, row.kind)
        out.append(bs.implied_vol(mp, S0, row.K, row.T, r, q, row.kind))
    return np.array(out)


def synthetic_market(S0=100.0, r=0.03, q=0.0, true: HestonParams | None = None,
                     maturities=(0.1, 0.25, 0.5, 1.0, 2.0),
                     moneyness=(0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2),
                     noise_iv_bp: float = 20.0, seed: int = 0):
    """Option quotes generated from known Heston parameters, plus bid/ask-like
    noise on implied volatility. OTM puts below the forward, OTM calls above
    (as traders quote them)."""
    import pandas as pd

    true = true or HestonParams(v0=0.035, kappa=2.0, theta=0.05, xi=0.6, rho=-0.7)
    rng = np.random.default_rng(seed)
    rows = []
    for T in maturities:
        F = S0 * np.exp((r - q) * T)
        Ks = F * np.asarray(moneyness)
        for K in Ks:
            kind = "put" if K < F else "call"
            p_true = price(S0, K, T, r, q, true, kind)
            iv = bs.implied_vol(p_true, S0, K, T, r, q, kind)
            iv_noisy = iv + rng.normal(0, noise_iv_bp * 1e-4)
            rows.append({"T": T, "K": K, "kind": kind,
                         "price": float(bs.price(S0, K, T, r, iv_noisy, q, kind)),
                         "true_iv": iv, "market_iv": iv_noisy})
    return pd.DataFrame(rows), true


def load_option_chain_csv(path, S0, r, q=0.0, min_price=0.05):
    """Load real quotes from a CSV with columns: K, T (years), bid, ask, kind.

    Uses the mid price, drops stale/illiquid quotes and keeps OTM options only.
    """
    import pandas as pd

    df = pd.read_csv(path)
    df["price"] = 0.5 * (df["bid"] + df["ask"])
    df = df[(df["bid"] > 0) & (df["price"] > min_price)]
    F = S0 * np.exp((r - q) * df["T"])
    otm = ((df["kind"] == "put") & (df["K"] < F)) | ((df["kind"] == "call") & (df["K"] >= F))
    df = df[otm].copy()
    # tighter spread -> more weight
    spread = (df["ask"] - df["bid"]).clip(lower=0.01)
    df["weight"] = 1 / np.sqrt(spread / spread.median())
    return df.reset_index(drop=True)
