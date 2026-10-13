import numpy as np
import pytest

from optpricing import (black_scholes as bs, heston, price_american_lsm,
                        price_binomial, price_european_mc, price_fd)

S0, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.2


def test_black_scholes_known_value_and_parity():
    c = bs.price(S0, K, T, R, SIG, kind="call")
    p = bs.price(S0, K, T, R, SIG, kind="put")
    assert c == pytest.approx(10.4506, abs=1e-4)          # textbook value
    assert c - p == pytest.approx(S0 - K * np.exp(-R * T), abs=1e-10)


def test_implied_vol_roundtrip():
    for sig in (0.1, 0.25, 0.6):
        pr = bs.price(S0, 110, 0.5, R, sig, kind="call")
        assert bs.implied_vol(pr, S0, 110, 0.5, R, kind="call") == pytest.approx(sig, abs=1e-8)


def test_greeks_match_finite_bumps():
    h = 1e-3
    g = bs.greeks(S0, K, T, R, SIG)
    up, dn = bs.price(S0 + h, K, T, R, SIG), bs.price(S0 - h, K, T, R, SIG)
    assert g["delta"] == pytest.approx((up - dn) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((up - 2 * bs.price(S0, K, T, R, SIG) + dn) / h ** 2, rel=1e-4)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_fd_european_matches_closed_form(kind):
    res = price_fd(S0, K, T, R, SIG, kind=kind, n_space=400, n_time=400)
    exact = bs.price(S0, K, T, R, SIG, kind=kind)
    assert res.price == pytest.approx(exact, abs=2e-3)
    assert res.delta == pytest.approx(bs.greeks(S0, K, T, R, SIG, kind=kind)["delta"], abs=1e-3)


def test_fd_second_order_convergence():
    exact = bs.price(S0, K, T, R, SIG)
    e1 = abs(price_fd(S0, K, T, R, SIG, n_space=200, n_time=200).price - exact)
    e2 = abs(price_fd(S0, K, T, R, SIG, n_space=400, n_time=400).price - exact)
    assert 3.5 < e1 / e2 < 4.5       # halving h divides the error by ~4


def test_american_put_fd_vs_binomial_and_bounds():
    am = price_fd(S0, K, T, R, SIG, kind="put", american=True, n_space=400, n_time=400).price
    tree = price_binomial(S0, K, T, R, SIG, kind="put", american=True, n_steps=2000)
    assert am == pytest.approx(tree, abs=3e-3)
    assert am > bs.price(S0, K, T, R, SIG, kind="put")    # early exercise has value


def test_american_call_without_dividends_equals_european():
    am = price_fd(S0, K, T, R, SIG, kind="call", american=True, n_space=400, n_time=400).price
    assert am == pytest.approx(bs.price(S0, K, T, R, SIG), abs=2e-3)


@pytest.mark.parametrize("method", ["plain", "antithetic", "control"])
def test_mc_within_confidence_interval(method):
    res = price_european_mc(S0, K, T, R, SIG, n_paths=200_000, method=method, seed=11)
    exact = bs.price(S0, K, T, R, SIG)
    assert abs(res.price - exact) < 4 * res.std_error


def test_control_variate_reduces_error():
    plain = price_european_mc(S0, K, T, R, SIG, n_paths=100_000, method="plain", seed=1)
    cv = price_european_mc(S0, K, T, R, SIG, n_paths=100_000, method="control", seed=1)
    assert cv.std_error < 0.6 * plain.std_error


def test_lsm_american_put_close_to_tree():
    lsm = price_american_lsm(S0, K, T, R, SIG, kind="put", n_paths=100_000, n_steps=50, seed=5)
    tree = price_binomial(S0, K, T, R, SIG, kind="put", american=True, n_steps=2000)
    assert abs(lsm.price - tree) < 0.06


def test_heston_reduces_to_black_scholes_without_vol_of_vol():
    p = heston.HestonParams(v0=SIG ** 2, kappa=2.0, theta=SIG ** 2, xi=1e-4, rho=0.0)
    Ks = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(heston.price(S0, Ks, T, R, 0.0, p),
                               bs.price(S0, Ks, T, R, SIG), atol=1e-4)


def test_heston_put_call_parity():
    p = heston.HestonParams(0.04, 1.5, 0.05, 0.5, -0.7)
    c = heston.price(S0, 105, 0.5, R, 0.01, p, "call")
    pu = heston.price(S0, 105, 0.5, R, 0.01, p, "put")
    assert c - pu == pytest.approx(S0 * np.exp(-0.01 * 0.5) - 105 * np.exp(-R * 0.5), abs=1e-8)


def test_heston_calibration_recovers_parameters():
    quotes, true = heston.synthetic_market(noise_iv_bp=0.0, seed=0)
    cal = heston.calibrate(quotes, 100.0, 0.03)
    np.testing.assert_allclose(cal.params.to_vector(), true.to_vector(), rtol=0.02, atol=1e-3)
    assert cal.rmse_iv_bp < 1.0
