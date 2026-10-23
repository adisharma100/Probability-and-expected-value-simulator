"""Core Monte Carlo engine for a repeated coin-flip betting game.

Game: each round you stake a fraction f of your CURRENT bankroll.
  - win  (prob p): bankroll *= 1 + f*b     (b = net odds, b=1 means even money)
  - lose (prob q): bankroll *= 1 - f
"""
import numpy as np


def edge(p: float, b: float = 1.0) -> float:
    """Expected profit per unit staked: E[X] = p*b - (1-p)."""
    return p * b - (1 - p)


def kelly_fraction(p: float, b: float = 1.0) -> float:
    """Kelly-optimal fraction: f* = (b*p - q) / b = edge / b."""
    return edge(p, b) / b


def theoretical_growth(f, p: float, b: float = 1.0):
    """Expected log-growth per bet: g(f) = p*ln(1+f*b) + q*ln(1-f)."""
    f = np.asarray(f, dtype=float)
    return p * np.log1p(f * b) + (1 - p) * np.log1p(-f)


def simulate(p, b, f, n_bets, n_paths, start=1.0, seed=0):
    """Simulate n_paths bankroll trajectories using a fixed fraction f.

    Returns an array of shape (n_paths, n_bets + 1); column 0 is the start.
    """
    rng = np.random.default_rng(seed)
    wins = rng.random((n_paths, n_bets)) < p
    multipliers = np.where(wins, 1 + f * b, 1 - f)
    paths = np.empty((n_paths, n_bets + 1))
    paths[:, 0] = start
    paths[:, 1:] = start * np.cumprod(multipliers, axis=1)
    return paths


def simulate_fixed_stake(p, b, stake, n_bets, n_paths, start=0.0, seed=0):
    """Flat betting (same dollar stake every time, no compounding)."""
    rng = np.random.default_rng(seed)
    wins = rng.random((n_paths, n_bets)) < p
    pnl = np.where(wins, stake * b, -stake)
    paths = np.empty((n_paths, n_bets + 1))
    paths[:, 0] = start
    paths[:, 1:] = start + np.cumsum(pnl, axis=1)
    return paths


def max_drawdown(paths):
    """Largest peak-to-trough fall (as a fraction) for each path."""
    running_peak = np.maximum.accumulate(paths, axis=1)
    return ((running_peak - paths) / running_peak).max(axis=1)


def summarize(paths, ruin_level=0.1):
    """Key statistics of final wealth across all simulated paths."""
    final = paths[:, -1]
    n_bets = paths.shape[1] - 1
    return {
        "mean_final": final.mean(),
        "median_final": np.median(final),
        "std_final": final.std(),
        "prob_loss": (final < paths[:, 0]).mean(),
        "prob_ruin": (paths.min(axis=1) < ruin_level * paths[:, 0]).mean(),
        "avg_max_drawdown": max_drawdown(paths).mean(),
        "growth_per_bet": (np.log(np.maximum(final, 1e-300) / paths[:, 0]) / n_bets).mean(),
    }
