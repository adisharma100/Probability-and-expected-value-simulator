# Probability & Expected-Value Simulator

Monte Carlo study of betting games, position sizing, the Kelly criterion and
simple options payoffs. Built with plain NumPy + Matplotlib.

## Run
```bash
pip install -r requirements.txt
python run_study.py        # prints tables, saves plots/ 
```

## Files
- `betting.py`: simulation engine (fixed-fraction and flat betting, Kelly formula, drawdown, summary stats)
- `options.py`: call/put/straddle payoffs, Black-Scholes price, Monte Carlo pricing with standard error
- `run_study.py`: four experiments that produce the plots and tables

## Studies and what they show
1. **Variance**: fair coin, flat $1 bets. EV stays 0 but the spread of outcomes grows like sqrt(n).
2. **Edge**: p = 0.50 to 0.60. Higher edge raises mean P&L, yet at p=0.52 you still lose money ~26% of the time over 200 bets.
3. **Kelly**: p=0.55, even money, so f* = edge/odds = 10%. Sweeping the bet fraction shows
   growth peaks at f*, falls past it, and hits zero at 2x Kelly. Half-Kelly gives up some growth for much smaller drawdowns.
4. **Options**: Monte Carlo call price (GBM) converges to Black-Scholes at rate 1/sqrt(N); straddle payoff and P&L distribution.

## Key results (from a run with 10,000 paths x 200 bets, p=0.55)
| Bet size | Median final | Mean final | P(loss) | P(ruin <10%) | Avg max drawdown |
|---|---|---|---|---|---|
| 0.5x Kelly | 2.12 | 2.71 | 14.5% | 0.0% | 43.5% |
| 1x Kelly | 2.72 | 7.34 | 26.0% | 2.7% | 70.1% |
| 2x Kelly | 0.97 | 68.39 | 53.4% | 40.5% | 93.4% |

Note how the **mean** at 2x Kelly is huge while the **median** is below the starting bankroll: a few lucky paths
drag the average up. Expected value is not the same as what a typical trader experiences.

## Interview talking points
- Kelly: f* = (bp - q)/b maximises expected log wealth; over-betting is far worse than under-betting.
- Variance drag: compounding growth ~ mean - variance/2, which is why large bets hurt.
- Mean vs median vs ruin: EV alone is a bad decision rule once bets compound.
- Monte Carlo error shrinks as 1/sqrt(N), so 100x more paths buys 10x more accuracy.

## Ideas to extend
- Unknown edge: bet with an estimated p and see how estimation error hurts (why practitioners use fractional Kelly)
- Multiple simultaneous bets, a drawdown-limit rule, or fat-tailed returns
