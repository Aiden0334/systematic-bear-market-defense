# systematic-bear-market-defense
XGBOOST Model-based downside protection for Bitcoin via liquidity-driven event detection

---

## Key Results

**This is not an alpha strategy.**
**This is a risk management model.**

The model detects liquidity-driven events (foreced liquidations, volatility explosions) and captures short-term mean reversion. 
It activates during bear markets and stays inactive during quiet bull markets. 

**Bear Market Defense Validated Across 5 Years (2021-2025):**

A. 2022 crash: BTC -64% (BnH) / ETH -68% (BnH) 
  - Strategy: **-9%** / **+14%**
B. 2025 bearish: BTC -6% (BnH) / ETH -11% (BnH)
  - Strategy: **+24%** / **+9%**

**Portfolio (BTC 20% BnH + Strategy 80%):**

A. CAGR: **20.4%** / MDD: **-12,2%** / Sharpe Ratio: **1.075**
B. Profitable years: **5/5** (2021-2025)

**Out of Sample Validation:**
  - Same model, same parameters applied to ETH
  - BTC CAGR 5.64% / ETH CAGR 5.48% - **0.16%p difference**
  - Confirmed liquidity-driven edge, not asset-specific overfitting

---

## Motivation

The initial goal was to develop an alpha strategy leveraging BTC 15-minute bar data. However, two critical issues were identified during validation: 

**1. Data Noise**
  - 15-min target autocorrelation: -0.05 (random noise level, discard)
  - 1 hour target autocorrelation: 0.59 (leverage available)
  - Unpredictable at short horizons
  - Switched to 4 hour targets: autocorrelation 0.81 (Strong momentum signal)
    **The reason why I chose 4 hour horizon.**

**2. Insufficient Samples in Bull Markets**
  - Liquidity events are rare during quiet bull markets
  - Trade count dropped to single digits (2023: 9 trades, 2024: 1 trade)
  - Statistical reliability could not be established for  alpha generation

**Conclusion**

Data confirmed that the real edge lies not in generating alpha during bull markets, but in **defending against losses during bear markets and market stress events.**

The strategy was repositioned as a **risk management model**:
  - Activates when liquidity events occur (forced liquidations, volatility explosions)
  - Stays inactive from quiet bull markets
  - Combined with BnH position for full market cycle coverage

--- 

## Model Overview

This model detects **liquidity-driven events** in cryptocurrency futures markets and captures short-term mean reversion by following price dislocations.

**Why liquidity events?**

The cryptocurrency futures market is highly leveraged. If prices move in a direction unfavorable to leveraged positions, a chain reaction of forced liquidations occurs and cause temporary but predictable price distortions. These price distortions quickly revert to normal levels, presenting trading opportunities. 

**Why inactive in bull markets?**

Liquidity events are rare when markets trend smoothly upward. Low signal frequency in bull markets is not a model failure — it reflects the absence of the specific conditions the model was designed to leverage.

---

## Methodology

### Data
- **Source**: Binance USDT-M Perpetual Futures
- **Assets**: BTCUSDT (primary), ETHUSDT (out of sample validation)
- **Timeframe**: 15-minute bars
- **Period**: 2019-09-25 ~ 2026-04-23 (~230,000 bars)
- **Features**: 100 features across momentum, volatility, volume, order flow, open interest, funding rate, long/short ratio, time **(removed smc features for final validation)**

> Raw data not included due to file size. See `src/config.py` for data collection setup.

### Target Definition
4-hour direction label (16 bars × 15 min) with ±0.8% threshold:
- `+1`: price rises ≥ +0.8% after 4 hours
- `0`: price stays within ±0.8%
- `-1`: price falls ≥ -0.8% after 4 hours

### Model
**XGBoost Classifier** — 3-class (down / sideways / up)

| Parameter | Value | Reason |
|-----------|-------|--------|
| n_estimators | 100 | Balance between speed and accuracy |
| max_depth | 4 | Prevent overfitting |
| learning_rate | 0.05 | Slow convergence, better generalization |
| subsample | 0.8 | Row subsampling, prevent overfitting |
| colsample_bytree | 0.8 | Feature subsampling, prevent overfitting |
| min_child_weight | 10 | Prevent learning sparse patterns |
| reg_alpha | 0.1 | L1 regularization |
| reg_lambda | 1.0 | L2 regularization |

**Entry**: Long signal when predicted up probability > 0.50
**Stop Loss**: Entry price − 2×ATR (dynamic, volatility-adjusted)
**Max Hold**: 64 bars (16 hours)
**Cooldown**: 16 bars (4 hours) after exit

**Why Long-only?**

Short signals showed inconsistent win rates across all walk-forward years. Validated by data, not assumption.

### Walk-Forward Validation

> Training: Rolling 2-year window
> Testing:  1-year out-of-sample
> Safety:   Skip if training data < 700 days

**Why rolling window (not expanding)?**

Cryptocurrency markets experience frequent distribution shifts. Older data can introduce noise rather than signal. Rolling window ensures the model learns from recent market regimes only.

**Why 700-day minimum?**

Exact 2 years = 730 days, but leap year variations cause 729-day windows. 700-day threshold provides margin while ensuring sufficient training data.

### Feature Engineering

100 features across 21 categories. Key categories:

| Category | Examples | Purpose |
|----------|---------|---------|
| Volatility | volatility_5m/15m/60m/240m | Detect volatility explosion |
| HL Range | hl_range_5m/15m/60m/240m | Measure price range (top importance: 0.162) |
| Liquidity Sweep | swept_high/low_15m/60m | Detect stop hunts |
| Volume | vol_spike_240m, vol_buy_pressure | Confirm event strength |
| Open Interest | oi_roc_15m, oi_zscore | Detect forced liquidations |
| Funding Rate | fr_level, fr_extreme | Measure market positioning stress |
| Swing VWAP | swing_vwap_dist, swing_vwap_dir | Liquidity-based price position |
| Time | time_us_session, time_hour_sin/cos | Session-based liquidity patterns |

> Full feature list: `models/feature_cols.json`
> Feature descriptions: `models/feature_description.md`

**Note on SMC features**: Smart Money Concept features (BOS, CHoCH, Order Blocks, FVG) were initially included as core liquidity event indicators. However, Binance API data quality issues (>80% NaN or noise, serious) led to their removal. **MDD improved significantly after removal.**

### Seed Robustness

| Metric | Value |
|--------|-------|
| Seeds tested | 10 (0, 1, 2, 3, 7, 42, 100, 2024, 9999, 31415) |
| Positive CAGR | 10/10 |
| Average CAGR | 10.84% |
| CAGR Std | 3.90% |

Low standard deviation confirms results are not dependent on a lucky random seed.

---

## 📈 Results

### BTC Strategy (5-Year Walk-Forward)

| Year | Return | Sharpe | MDD | Trades | Win Rate | BnH |
|------|--------|--------|-----|--------|----------|-----|
| 2021 | 0.00% (blocked) | — | — | 0 | — | +64.2% |
| 2022 | **-9.0%** | 0.441 | -22.1% | 64 | 0.469 | -64.2% |
| 2023 | +13.8% | N/A* | -2.1% | 9 | 0.667 | +156.1% |
| 2024 | +9.7% | N/A* | -6.5% | 1 | 1.000 | +118.2% |
| 2025 | **+24.0%** | 0.762 | -8.3% | 25 | 0.480 | -5.6% |

> Year 2026 is delisted because of low test samples. 
> *Sharpe marked N/A when trade count < 30 (statistically unreliable)

### ETH Out-of-Sample (Same Model, Same Parameters)

| Year | Return | MDD | Trades | BnH |
|------|--------|-----|--------|-----|
| 2021 | 0.00% (blocked) | — | 0 | +409.1% |
| 2022 | **+14.1%** | -47.8% | 228 | -67.6% |
| 2023 | +12.2% | 0.0% | 1 | +92.5% |
| 2024 | +23.1% | -6.5% | 11 | +46.3% |
| 2025 | **+9.4%** | -24.9% | 129 | -11.4% |

### Risk Management Validation

| Scenario | BTC BnH | BTC Strategy | ETH BnH | ETH Strategy |
|----------|---------|-------------|---------|-------------|
| 2022 Crash | -64.2% | **-9.0%** | -67.6% | **+14.1%** |
| 2025 Bear | -5.6% | **+24.0%** | -11.4% | **+9.4%** |

> **Both assets. Same model. Same parameters. Consistent bear market defense.**

---

## Failed Attempts

Honest record of what was tried and why it was abandoned.

| Attempt | Result | Reason Abandoned |
|---------|--------|-----------------|
| 15-min target | Failed | Lag-1 AC -0.05, pure noise |
| Long + Short | Failed | Short signals inconsistent across all years |
| HMM Regime Filter (3-state) | Failed | Distribution shift, model misclassified BTC regimes |
| HMM Persistence Feature | Failed | Information overlap, degraded existing model |
| Dynamic Allocation (MA50/200) | Failed | Lagging signal, CAGR dropped to 8.3% |
| SMC Features | Removed | Binance API data quality > 80% NaN/noise | **Key Features for alpha strategy**
| ETH Portfolio Expansion | Abandoned | ETH strategy MDD -47.8% (2022), operationally unreliable |
| Expanding Window Walk-Forward | Rejected | Distribution shift in crypto markets makes old data noise |

---

## Limitations

**1. Bull Market Underperformance**
The strategy significantly underperforms BnH during strong bull markets (2023: +13.8% vs BnH +156.1%). This is by design — the model only activates during liquidity events, which are rare in trending bull markets.

**2. Low Trade Count in Certain Years**
2023 (9 trades) and 2024 (1 trade) have insufficient samples for statistically reliable Sharpe ratios. Sharpe marked N/A for these years.

**3. SMC Feature Gap**
Smart Money Concept features were theoretically the most relevant for liquidity event detection but had to be removed due to Binance API data quality issues. This remains the biggest limitation of the current implementation.

**4. No Live Trading Validation**
All results are based on backtests. Slippage, latency, and execution costs in live trading may differ from simulated results.

**5. Single Exchange Dependency**
Data sourced exclusively from Binance USDT-M Futures. Cross-exchange liquidity dynamics not captured.

---

## How to Run

### Requirements
```bash
pip install -r requirements.txt
```

### Data Preparation
Raw data is not included due to file size constraints (~500MB).

### Results
Pre-computed results available in `validator & results/`:
  - `6y_walkforward_safe_results.csv` (BTC 6 year walk-forward)
  - `ETH_walkforward_results.csv` (ETH out of sample validation)

---

## Repository Structure

```
systematic-bear-market-defense/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   ├── make_targets_btc.py    ← Target label generation
│   └── config.py              ← Data collection configuration
├── models/
│   └── feature_cols.json      ← 100 selected features
└── validator & results/
    ├── 6y_walkforward_safe_results.csv
    ├── ETH_walkforward_results.csv
    ├── walkforward_btc1.py    ← BTC 6-year walk-forward
    └── walkforward_eth1.py    ← ETH out-of-sample validation
```

---

## Lessons Learned

**1. Data quality over model complexity**
SMC features were theoretically superior but practically useless due to data quality. Removing them improved MDD significantly. Clean simple features outperformed complex noisy ones.

**2. Let data redefine the objective**
Started with alpha generation. Data showed the real edge was in bear market defense. Repositioning the objective led to a more honest and validated strategy.

**3. Walk-forward discipline**
Every decision was validated through walk-forward testing, not in-sample optimization. Rolling window over expanding window because crypto markets shift frequently.

**4. Seed robustness is non-negotiable**
Single seed results are anecdotal. 10-seed validation confirmed CAGR std of 3.90% — low enough to trust the signal over luck.

**5. Honest reporting builds credibility**
Strong bull market underperformance is reported without hiding. Failed attempts are documented. Statistically unreliable Sharpe ratios are marked N/A. Transparency is the point.

---

## Future Work

**Project 2 — Statistical Mean Reversion Alpha Strategy Modeling**
  - Bollinger Band ±2σ breakout with several regime filters.
  - Multi-timeframe: A day or 4-hour signal + 1 hour entry (testing...)
  - CLT-based justification (window ≥ 30).

**Project 3 — Option Pricing Modeling**
  - Black-Scholes Model leveraging deep-learning model

**Improvement Areas**
  - Higher quality liquidity data (order book, on-chain)
  - SMC feature re-validation with better data source
  - Cross-exchange data integration

---

**Data source: Binance USDT-M Perpetual Futures**
**Backtest period: 2021–2025 (5-year walk-forward)**
**Out-of-sample: ETH validation with identical model setup**


