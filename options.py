"""Simple options payoffs and Monte Carlo pricing vs Black-Scholes."""
import math
import numpy as np


def call_payoff(S, K):
    return np.maximum(S - K, 0.0)


def put_payoff(S, K):
    return np.maximum(K - S, 0.0)


def straddle_payoff(S, K, premium=0.0):
    """Long call + long put at the same strike, minus premium paid."""
    return call_payoff(S, K) + put_payoff(S, K) - premium


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_call(S0, K, r, sigma, T):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def mc_call_price(S0, K, r, sigma, T, n_paths, seed=0):
    """Price a European call by simulating terminal prices under GBM.

    S_T = S0 * exp((r - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0,1)
    Returns (price, standard_error).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
    discounted = math.exp(-r * T) * call_payoff(ST, K)
    return discounted.mean(), discounted.std(ddof=1) / math.sqrt(n_paths)
