# Quant Research / Quant Developer Track

This track uses the same rigor as ML interviews, with extra emphasis on data leakage, assumptions, execution realism, and monitoring.

## Core topics

- Probability, statistics, hypothesis testing, time series
- Feature engineering and signal validation
- Walk-forward validation and purged/embargoed splits
- Transaction costs, slippage, liquidity, capacity, turnover
- Sharpe, drawdown, tail risk, exposure decomposition
- Python data structures; optionally C++/performance fundamentals
- Event-driven system design, queues, latency, correctness

## Research answer structure

Problem -> assumptions -> baseline -> data and leakage controls -> model/signal -> validation -> execution realism -> results -> failure modes -> monitoring.

## Drills

1. Design a mean-reversion strategy; identify every leakage path.
2. A backtest Sharpe is 2.5 but live PnL is flat. Debug it.
3. Build a market-data feature pipeline for millions of ticks/day.
4. Explain Kalman filtering intuitively and contrast it with rolling regression.
5. Implement an online rolling z-score, top-k, and bounded-memory event buffer.
