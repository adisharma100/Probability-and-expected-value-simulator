"""Runs all experiments and saves plots to ./plots. Usage: python run_study.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from betting import (edge, kelly_fraction, theoretical_growth, simulate,
                     simulate_fixed_stake, summarize)
from options import (black_scholes_call, mc_call_price, call_payoff,
                     put_payoff, straddle_payoff)

N_PATHS, N_BETS = 10_000, 200
PLOTS = "plots/"


def study_1_variance():
    """Fair game: EV is zero, but outcomes spread out as sqrt(n)."""
    paths = simulate_fixed_stake(0.5, 1.0, 1.0, N_BETS, N_PATHS)
    x = np.arange(N_BETS + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x, paths[:200].T, alpha=0.08, color="steelblue")
    plt.plot(x, paths.mean(axis=0), color="black", lw=2, label="mean (= EV = 0)")
    plt.plot(x, np.sqrt(x), "r--", label=r"$\pm\sqrt{n}$ (1 std dev)")
    plt.plot(x, -np.sqrt(x), "r--")
    plt.title("Study 1: fair coin, flat $1 bets: same EV, growing variance")
    plt.xlabel("bets"); plt.ylabel("cumulative P&L"); plt.legend()
    plt.savefig(PLOTS + "1_variance.png", dpi=130, bbox_inches="tight"); plt.close()
    print(f"[1] final P&L mean={paths[:, -1].mean():+.2f}, "
          f"std={paths[:, -1].std():.2f}  (theory: 0, {np.sqrt(N_BETS):.2f})")


def study_2_edge():
    """Bigger edge -> higher EV, but luck still dominates short horizons."""
    print("\n[2] Edge study (flat $1 bets, 200 bets)")
    print(f"{'p':>5} {'edge/bet':>9} {'mean P&L':>9} {'std':>7} {'P(loss)':>8}")
    plt.figure(figsize=(8, 5))
    for p in [0.50, 0.52, 0.55, 0.60]:
        final = simulate_fixed_stake(p, 1.0, 1.0, N_BETS, N_PATHS)[:, -1]
        print(f"{p:>5.2f} {edge(p):>9.3f} {final.mean():>9.2f} "
              f"{final.std():>7.2f} {(final < 0).mean():>8.1%}")
        plt.hist(final, bins=50, alpha=0.5, label=f"p={p:.2f}")
    plt.title("Study 2: distribution of 200-bet P&L by win probability")
    plt.xlabel("final P&L"); plt.ylabel("paths"); plt.legend()
    plt.savefig(PLOTS + "2_edge.png", dpi=130, bbox_inches="tight"); plt.close()


def study_3_kelly(p=0.55, b=1.0):
    """Sweep bet fraction: simulated growth vs the Kelly formula."""
    fk = kelly_fraction(p, b)
    print(f"\n[3] Kelly study: p={p}, b={b}, edge={edge(p, b):.2f}, Kelly f*={fk:.3f}")
    fs = np.linspace(0.0, 0.5, 26)
    sim_growth, ruin, med = [], [], []
    for f in fs:
        s = summarize(simulate(p, b, f, N_BETS, N_PATHS, seed=1))
        sim_growth.append(s["growth_per_bet"]); ruin.append(s["prob_ruin"])
        med.append(s["median_final"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(fs, theoretical_growth(fs, p, b), label="theory g(f)")
    ax[0].plot(fs, sim_growth, "o", ms=4, label="simulated")
    ax[0].axvline(fk, color="green", ls="--", label=f"Kelly f*={fk:.2f}")
    ax[0].axvline(2 * fk, color="red", ls=":", label="2x Kelly (growth = 0)")
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_xlabel("fraction of bankroll bet (f)")
    ax[0].set_ylabel("log-growth per bet"); ax[0].legend()
    ax[0].set_title("Growth rate peaks at Kelly, goes negative beyond 2x")
    ax[1].plot(fs, ruin, color="crimson")
    ax[1].axvline(fk, color="green", ls="--")
    ax[1].set_xlabel("fraction of bankroll bet (f)")
    ax[1].set_ylabel("P(bankroll ever < 10% of start)")
    ax[1].set_title("Risk of ruin rises sharply with over-betting")
    plt.savefig(PLOTS + "3_kelly_growth.png", dpi=130, bbox_inches="tight"); plt.close()

    print(f"{'f':>10} {'median':>8} {'mean':>8} {'P(loss)':>8} {'P(ruin)':>8} {'avg MDD':>8}")
    plt.figure(figsize=(9, 5.5))
    for label, f, c in [("0.5x Kelly", fk / 2, "tab:blue"),
                        ("1x Kelly", fk, "tab:green"),
                        ("2x Kelly", 2 * fk, "tab:red")]:
        paths = simulate(p, b, f, N_BETS, N_PATHS, seed=2)
        s = summarize(paths)
        print(f"{label:>10} {s['median_final']:>8.2f} {s['mean_final']:>8.2f} "
              f"{s['prob_loss']:>8.1%} {s['prob_ruin']:>8.1%} {s['avg_max_drawdown']:>8.1%}")
        plt.plot(np.median(paths, axis=0), color=c, lw=2, label=f"{label} median")
        plt.plot(np.percentile(paths, 10, axis=0), color=c, lw=1, ls="--")
        plt.plot(np.percentile(paths, 90, axis=0), color=c, lw=1, ls="--")
    plt.yscale("log"); plt.xlabel("bets"); plt.ylabel("bankroll (log scale)")
    plt.title("Median and 10th/90th percentile bankroll by bet size")
    plt.legend(); plt.savefig(PLOTS + "3_kelly_fan.png", dpi=130, bbox_inches="tight"); plt.close()


def study_4_options():
    S0, K, r, sigma, T = 100, 100, 0.03, 0.20, 1.0
    bs = black_scholes_call(S0, K, r, sigma, T)
    print(f"\n[4] Options: Black-Scholes call = {bs:.4f}")
    ns = [100, 1_000, 10_000, 100_000, 1_000_000]
    prices, errs = [], []
    for n in ns:
        pr, se = mc_call_price(S0, K, r, sigma, T, n)
        prices.append(pr); errs.append(se)
        print(f"   N={n:>9,}  MC={pr:7.4f}  std err={se:.4f}  error={pr - bs:+.4f}")
    plt.figure(figsize=(8, 5))
    plt.errorbar(ns, prices, yerr=1.96 * np.array(errs), fmt="o-", capsize=4,
                 label="MC price +/- 95% CI")
    plt.axhline(bs, color="green", ls="--", label="Black-Scholes")
    plt.xscale("log"); plt.xlabel("number of simulated paths"); plt.ylabel("call price")
    plt.title("Monte Carlo converges at rate 1/sqrt(N)"); plt.legend()
    plt.savefig(PLOTS + "4_option_convergence.png", dpi=130, bbox_inches="tight"); plt.close()

    # payoff diagram at expiry
    S = np.linspace(60, 140, 200)
    put_bs = bs - S0 + K * np.exp(-r * T)   # put-call parity
    premium = bs + put_bs
    plt.figure(figsize=(8, 5))
    plt.plot(S, call_payoff(S, K) - bs, label="long call")
    plt.plot(S, put_payoff(S, K) - put_bs, label="long put")
    plt.plot(S, straddle_payoff(S, K, premium), label="long straddle")
    plt.axhline(0, color="gray", lw=0.8)
    plt.xlabel("price at expiry"); plt.ylabel("profit at expiry")
    plt.title("Option payoff diagrams"); plt.legend()
    plt.savefig(PLOTS + "4_option_payoffs.png", dpi=130, bbox_inches="tight"); plt.close()

    # straddle P&L distribution by simulation
    rng = np.random.default_rng(5)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * rng.standard_normal(200_000))
    pnl = straddle_payoff(ST, K, premium * np.exp(r * T))  # premium grown at r to expiry
    print(f"   Straddle (premium {premium:.2f}): mean P&L={pnl.mean():+.3f}, "
          f"P(profit)={(pnl > 0).mean():.1%}  <- fair price: EV ~ 0, but wins well under half the time")


if __name__ == "__main__":
    study_1_variance()
    study_2_edge()
    study_3_kelly()
    study_4_options()
    print("\nPlots saved in ./plots")
