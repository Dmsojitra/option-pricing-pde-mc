"""Run every experiment: accuracy, convergence, runtime, American options, Greeks
and Heston calibration. Writes figures/ and results/benchmark.json.

    python scripts/run_all.py
    python scripts/run_all.py --chain data/my_chain.csv --spot 450 --rate 0.04
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optpricing import black_scholes as bs  # noqa: E402
from optpricing import heston, plotstyle  # noqa: E402
from optpricing import (price_american_lsm, price_binomial,  # noqa: E402
                        price_european_mc, price_fd)

FIG = ROOT / "figures"
RES = ROOT / "results"
S0, K, T, R, SIG, Q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
C = plotstyle.SERIES


def timed(f, *a, repeat=1, **kw):
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = f(*a, **kw)
    return out, (time.perf_counter() - t0) / repeat


def main(chain=None, spot=None, rate=0.03):
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    plotstyle.apply()
    out: dict = {"setup": dict(S0=S0, K=K, T=T, r=R, sigma=SIG, q=Q)}

    # ------------------------------------------------ 1. European call: three methods
    exact = float(bs.price(S0, K, T, R, SIG, Q, "call"))
    fd, t_fd = timed(price_fd, S0, K, T, R, SIG, Q, "call", n_space=400, n_time=400, repeat=3)
    mc, t_mc = timed(price_european_mc, S0, K, T, R, SIG, Q, "call", 1_000_000, "control", seed=42)
    out["european_call"] = {
        "exact": exact,
        "finite_difference": {"price": fd.price, "abs_error": abs(fd.price - exact), "seconds": t_fd},
        "monte_carlo_control_1M": {"price": mc.price, "std_error": mc.std_error,
                                   "abs_error": abs(mc.price - exact), "seconds": t_mc},
    }

    # ------------------------------------------------ 2. FD convergence (log-log)
    Ns = [50, 100, 200, 400, 800, 1600]
    errs_cn, errs_cn_raw, errs_imp = [], [], []
    for n in Ns:
        errs_cn.append(abs(price_fd(S0, K, T, R, SIG, Q, "call", n_space=n, n_time=n).price - exact))
        errs_cn_raw.append(abs(price_fd(S0, K, T, R, SIG, Q, "call", n_space=n, n_time=n,
                                        rannacher_steps=0).price - exact))
        errs_imp.append(abs(price_fd(S0, K, T, R, SIG, Q, "call", n_space=n, n_time=n,
                                     theta=1.0).price - exact))
    slope_cn = np.polyfit(np.log(Ns[2:]), np.log(errs_cn[2:]), 1)[0]
    slope_imp = np.polyfit(np.log(Ns[2:]), np.log(errs_imp[2:]), 1)[0]
    out["fd_convergence"] = {"N": Ns, "cn_rannacher": errs_cn, "cn_plain": errs_cn_raw,
                             "implicit": errs_imp, "order_cn": -slope_cn, "order_implicit": -slope_imp}

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(Ns, errs_cn, "o-", color=C[0], label=f"Crank–Nicolson + Rannacher (order {-slope_cn:.2f})")
    ax.loglog(Ns, errs_imp, "s-", color=C[1], label=f"fully implicit (order {-slope_imp:.2f})")
    ref = errs_cn[2] * (np.array(Ns) / Ns[2]) ** -2.0
    ax.loglog(Ns, ref, ":", color=plotstyle.TEXT_2, label="reference slope −2")
    ax.set(xlabel="grid size N (space = time steps)", ylabel="|price error|",
           title="Finite-difference convergence, European call")
    ax.legend()
    fig.savefig(FIG / "01_fd_convergence.png")
    plt.close(fig)

    # ------------------------------------------------ 3. MC convergence + variance reduction
    paths = [1_000, 4_000, 16_000, 64_000, 256_000, 1_024_000]
    mc_curves = {}
    for method in ("plain", "antithetic", "control"):
        se = [price_european_mc(S0, K, T, R, SIG, Q, "call", n, method, seed=7).std_error
              for n in paths]
        mc_curves[method] = se
    out["mc_std_error"] = {"paths": paths, **mc_curves,
                           "variance_reduction_factor_control": (mc_curves["plain"][-1] /
                                                                 mc_curves["control"][-1]) ** 2,
                           "variance_reduction_factor_antithetic": (mc_curves["plain"][-1] /
                                                                    mc_curves["antithetic"][-1]) ** 2}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (method, se), col, mk in zip(mc_curves.items(), C, "os^"):
        ax.loglog(paths, se, marker=mk, color=col, label=method.replace("control", "control variate"))
    ax.loglog(paths, mc_curves["plain"][0] * (np.array(paths) / paths[0]) ** -0.5, ":",
              color=plotstyle.TEXT_2, label="reference slope −1/2")
    ax.set(xlabel="number of paths", ylabel="standard error",
           title="Monte Carlo error and variance reduction")
    ax.legend()
    fig.savefig(FIG / "02_mc_convergence.png")
    plt.close(fig)

    # ------------------------------------------------ 4. American put: FD vs tree vs LSM
    ref_tree, t_tree = timed(price_binomial, S0, K, T, R, SIG, Q, "put", True, 5000)
    am_fd, t_amfd = timed(price_fd, S0, K, T, R, SIG, Q, "put", True, n_space=800, n_time=800)
    lsm, t_lsm = timed(price_american_lsm, S0, K, T, R, SIG, Q, "put", 200_000, 100, seed=3)
    euro_put = float(bs.price(S0, K, T, R, SIG, Q, "put"))
    out["american_put"] = {
        "binomial_5000_steps": {"price": ref_tree, "seconds": t_tree},
        "finite_difference_800": {"price": am_fd.price, "diff_vs_tree": am_fd.price - ref_tree,
                                  "seconds": t_amfd},
        "lsm_200k_paths_100_dates": {"price": lsm.price, "std_error": lsm.std_error,
                                     "diff_vs_tree": lsm.price - ref_tree, "seconds": t_lsm},
        "european_put": euro_put,
        "early_exercise_premium": ref_tree - euro_put,
    }

    S_plot = np.linspace(60, 140, 200)
    am_grid = price_fd(S0, K, T, R, SIG, Q, "put", True, n_space=800, n_time=800)
    am_vals = np.interp(S_plot, am_grid.S_grid, am_grid.V_grid)
    eu_vals = bs.price(S_plot, K, T, R, SIG, Q, "put")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(S_plot, am_vals, color=C[0], label="American put (FD)")
    axes[0].plot(S_plot, eu_vals, color=C[1], ls="--", label="European put (exact)")
    axes[0].plot(S_plot, np.maximum(K - S_plot, 0), color=plotstyle.TEXT_2, lw=1, ls=":",
                 label="payoff K − S")
    axes[0].set(xlabel="stock price S", ylabel="option value",
                title="American vs European put (T = 1 y)")
    axes[0].legend()
    b = am_grid.exercise_boundary
    axes[1].plot(T - b[:, 0], b[:, 1], color=C[0])
    axes[1].fill_between(T - b[:, 0], 0, b[:, 1], color=C[0], alpha=0.12, lw=0)
    axes[1].text(0.5, 0.1, "exercise now", transform=axes[1].transAxes, color=plotstyle.TEXT)
    axes[1].text(0.25, 0.8, "hold", transform=axes[1].transAxes, color=plotstyle.TEXT)
    axes[1].set(xlabel="calendar time t [years]", ylabel="critical price S*(t)",
                ylim=(b[:, 1].min() - 5, K + 2), title="Early-exercise boundary (from FD)")
    fig.savefig(FIG / "03_american_put.png")
    plt.close(fig)

    # ------------------------------------------------ 5. Greeks: FD vs analytic
    S_list = np.linspace(70, 130, 25)
    d_fd, g_fd = [], []
    for s in S_list:
        r_ = price_fd(s, K, T, R, SIG, Q, "call", n_space=400, n_time=400)
        d_fd.append(r_.delta)
        g_fd.append(r_.gamma)
    gk = bs.greeks(S_list, K, T, R, SIG, Q, "call")
    out["greeks_max_abs_error"] = {"delta": float(np.max(np.abs(np.array(d_fd) - gk["delta"]))),
                                   "gamma": float(np.max(np.abs(np.array(g_fd) - gk["gamma"])))}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    Sd = np.linspace(70, 130, 300)
    gkd = bs.greeks(Sd, K, T, R, SIG, Q, "call")
    for ax, key, vals in ((axes[0], "delta", d_fd), (axes[1], "gamma", g_fd)):
        ax.plot(Sd, gkd[key], color=C[1], label="analytic")
        ax.plot(S_list, vals, "o", ms=5, color=C[0], label="finite difference")
        ax.set(xlabel="stock price S", ylabel=key, title=f"Call {key}: PDE vs closed form")
        ax.legend()
    fig.savefig(FIG / "04_greeks.png")
    plt.close(fig)

    # ------------------------------------------------ 5b. Why Rannacher start-up matters
    # Fine space grid + coarse time steps: plain Crank-Nicolson rings at the kink.
    Tg = 0.25
    plain = price_fd(S0, K, Tg, R, SIG, Q, "call", n_space=1000, n_time=25, rannacher_steps=0)
    smooth = price_fd(S0, K, Tg, R, SIG, Q, "call", n_space=1000, n_time=25, rannacher_steps=4)
    gamma_exact = float(bs.greeks(S0, K, Tg, R, SIG, Q, "call")["gamma"])
    out["rannacher_gamma_at_strike"] = {"exact": gamma_exact, "crank_nicolson": plain.gamma,
                                        "crank_nicolson_rannacher": smooth.gamma}

    def gamma_curve(res):
        x = np.log(res.S_grid)
        dx = x[1] - x[0]
        Vx = np.gradient(res.V_grid, dx)
        Vxx = np.gradient(Vx, dx)
        return (Vxx - Vx) / res.S_grid ** 2

    win = (plain.S_grid > 85) & (plain.S_grid < 115)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(plain.S_grid[win], gamma_curve(plain)[win], color=C[1], lw=1.2,
            label="Crank–Nicolson (oscillates)")
    ax.plot(smooth.S_grid[win], gamma_curve(smooth)[win], color=C[0], lw=2,
            label="Crank–Nicolson + Rannacher")
    Sg = plain.S_grid[win]
    ax.plot(Sg, bs.greeks(Sg, K, Tg, R, SIG, Q, "call")["gamma"], color=plotstyle.TEXT,
            ls=":", lw=1.5, label="exact")
    ax.set(xlabel="stock price S", ylabel="gamma", ylim=(-0.15, 0.2),
           title="Gamma near the strike: why the Rannacher start-up matters")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "06_rannacher_gamma.png")
    plt.close(fig)

    # ------------------------------------------------ 6. Heston calibration
    if chain:
        import pandas as pd  # noqa: F401
        quotes = heston.load_option_chain_csv(chain, spot, rate)
        s0, r_ = spot, rate
        true = None
    else:
        quotes, true = heston.synthetic_market(S0=100.0, r=0.03)
        s0, r_ = 100.0, 0.03
    cal, t_cal = timed(heston.calibrate, quotes, s0, r_)
    quotes["model_iv"] = heston.model_implied_vols(quotes, s0, r_, 0.0, cal.params)
    if "market_iv" not in quotes:
        quotes["market_iv"] = [bs.implied_vol(row.price, s0, row.K, row.T, r_, 0.0, row.kind)
                               for row in quotes.itertuples()]
    out["heston_calibration"] = {
        "fitted": dict(zip(heston.HestonParams.NAMES, map(float, cal.params.to_vector()))),
        "true": dict(zip(heston.HestonParams.NAMES, map(float, true.to_vector()))) if true else None,
        "iv_rmse_bp": cal.rmse_iv_bp, "n_evals": cal.n_evals, "seconds": t_cal,
        "feller_satisfied": bool(cal.params.feller()), "n_quotes": len(quotes),
    }

    mats = sorted(quotes["T"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for col, Tm in zip(C, mats):
        g = quotes[quotes["T"] == Tm].sort_values("K")
        fwd = s0 * np.exp(r_ * Tm)
        ax.plot(g.K / fwd, 100 * g.market_iv, "o", ms=5, color=col)
        ax.plot(g.K / fwd, 100 * g.model_iv, "-", color=col, label=f"T = {Tm:g} y")
    ax.set(xlabel="moneyness K / F", ylabel="implied volatility [%]",
           title=f"Heston calibration: dots = market, lines = model (RMSE {cal.rmse_iv_bp:.0f} bp)")
    ax.legend(title="maturity", ncols=2)
    fig.savefig(FIG / "05_heston_smile.png")
    plt.close(fig)

    with open(RES / "benchmark.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=lambda x: round(float(x), 6)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", help="CSV with K, T, bid, ask, kind (real option chain)")
    ap.add_argument("--spot", type=float, help="spot price for --chain")
    ap.add_argument("--rate", type=float, default=0.03, help="risk-free rate for --chain")
    a = ap.parse_args()
    if a.chain and a.spot is None:
        ap.error("--spot is required with --chain")
    main(a.chain, a.spot, a.rate)
