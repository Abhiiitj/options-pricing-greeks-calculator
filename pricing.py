from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class OptionResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def black_scholes(spot: float, strike: float, time_to_expiry: float, rate: float,
                  volatility: float, option_type: str = "call", dividend_yield: float = 0.0) -> OptionResult:
    """Price a European option and calculate Greeks; rates, volatility, and time are annualized."""
    if spot <= 0 or strike <= 0:
        raise ValueError("Spot and strike must be positive.")
    if time_to_expiry < 0:
        raise ValueError("Time to expiry cannot be negative.")
    if volatility < 0:
        raise ValueError("Volatility cannot be negative.")
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if time_to_expiry == 0 or volatility == 0:
        return _deterministic_value(spot, strike, time_to_expiry, rate, volatility, kind, dividend_yield)

    t, sigma = time_to_expiry, volatility
    root_t = sqrt(t)
    d1 = (log(spot / strike) + (rate - dividend_yield + 0.5 * sigma * sigma) * t) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    discount_q, discount_r = exp(-dividend_yield * t), exp(-rate * t)
    pdf = norm.pdf(d1)
    gamma = discount_q * pdf / (spot * sigma * root_t)
    vega = spot * discount_q * pdf * root_t / 100.0

    if kind == "call":
        price = spot * discount_q * norm.cdf(d1) - strike * discount_r * norm.cdf(d2)
        delta = discount_q * norm.cdf(d1)
        theta = (-spot * discount_q * pdf * sigma / (2 * root_t)
                 - rate * strike * discount_r * norm.cdf(d2)
                 + dividend_yield * spot * discount_q * norm.cdf(d1)) / 365.0
        rho = strike * t * discount_r * norm.cdf(d2) / 100.0
    else:
        price = strike * discount_r * norm.cdf(-d2) - spot * discount_q * norm.cdf(-d1)
        delta = discount_q * (norm.cdf(d1) - 1)
        theta = (-spot * discount_q * pdf * sigma / (2 * root_t)
                 + rate * strike * discount_r * norm.cdf(-d2)
                 - dividend_yield * spot * discount_q * norm.cdf(-d1)) / 365.0
        rho = -strike * t * discount_r * norm.cdf(-d2) / 100.0
    return OptionResult(float(price), float(delta), float(gamma), float(vega), float(theta), float(rho))


def _deterministic_value(spot: float, strike: float, t: float, rate: float, sigma: float,
                         kind: str, q: float) -> OptionResult:
    """Boundary behavior at expiry or zero volatility (Greeks at the kink are undefined)."""
    forward_spot, discounted_strike = spot * exp(-q * t), strike * exp(-rate * t)
    sign = 1 if kind == "call" else -1
    intrinsic = sign * (forward_spot - discounted_strike)
    price = max(intrinsic, 0.0)
    # At expiry, report directional delta away from the strike; at the kink use 0.5 convention.
    if t == 0:
        moneyness = spot - strike
        if abs(moneyness) < 1e-12:
            delta = sign * 0.5
        else:
            delta = sign * float(moneyness > 0)
        return OptionResult(price, delta, 0.0, 0.0, 0.0, 0.0)
    if abs(intrinsic) < 1e-12:
        delta = sign * 0.5 * exp(-q * t)
    else:
        delta = sign * exp(-q * t) * float(intrinsic > 0)
    return OptionResult(price, delta, 0.0, 0.0, 0.0, 0.0)


def implied_volatility(market_price: float, spot: float, strike: float, time_to_expiry: float,
                       rate: float, option_type: str = "call", dividend_yield: float = 0.0) -> float:
    """Solve for annualized volatility using a bracketed Brent root finder."""
    if market_price < 0:
        raise ValueError("Market price cannot be negative.")
    if time_to_expiry <= 0:
        raise ValueError("Implied volatility requires positive time to expiry.")
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    lower = max(spot * exp(-dividend_yield * time_to_expiry) - strike * exp(-rate * time_to_expiry), 0.0) if kind == "call" else max(strike * exp(-rate * time_to_expiry) - spot * exp(-dividend_yield * time_to_expiry), 0.0)
    upper = spot * exp(-dividend_yield * time_to_expiry) if kind == "call" else strike * exp(-rate * time_to_expiry)
    if market_price < lower - 1e-10 or market_price >= upper:
        raise ValueError(f"Price must be between no-arbitrage bounds {lower:.6g} and {upper:.6g}.")
    objective = lambda vol: black_scholes(spot, strike, time_to_expiry, rate, vol, kind, dividend_yield).price - market_price
    return float(brentq(objective, 1e-8, 10.0, xtol=1e-12))


def put_call_parity_residual(spot: float, strike: float, time_to_expiry: float, rate: float,
                             call_price: float, put_price: float, dividend_yield: float = 0.0) -> float:
    """Return C - P - (S*e^(-qT) - K*e^(-rT)); zero means parity holds."""
    return call_price - put_price - (spot * exp(-dividend_yield * time_to_expiry) - strike * exp(-rate * time_to_expiry))


def finite_difference_delta(spot: float, strike: float, time_to_expiry: float, rate: float,
                            volatility: float, option_type: str = "call", dividend_yield: float = 0.0,
                            step: float = 1e-3) -> float:
    """Central-difference delta estimate for checking the analytical Greek."""
    if step <= 0 or spot <= step:
        raise ValueError("step must be positive and smaller than spot.")
    up = black_scholes(spot + step, strike, time_to_expiry, rate, volatility, option_type, dividend_yield).price
    down = black_scholes(spot - step, strike, time_to_expiry, rate, volatility, option_type, dividend_yield).price
    return (up - down) / (2.0 * step)
