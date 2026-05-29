# Systematic Bear Market Defense via XGBoost
> Downside Protection for Bitcoin Holdings via Liquidity-Driven Event Detection

---

## Key Results

**This is not an alpha strategy. This is a risk management model.**

The model detects liquidity-driven events (forced liquidations, volatility explosions) and captures short-term mean reversion. It activates during bear markets and stays inactive during quiet bull markets.

**Bear market defense validated across 5 years (2021–2025):**

- 2022 crash: BTC (BnH) -64.2% → Strategy    **-11.69%**
              / ETH (BnH) -67.6% → Strategy  **+14.07%**
- 2025 bear:  BTC (BnH) -5.6% → Strategy     **+23.97%**
              / ETH (BnH) -11.4% → Strategy  **+9.36%**

**Out of sample validation (ETH):**
- Same model, same parameters applied to ETH without modification
- BTC CAGR 5.64% / ETH CAGR 5.48% — **0.16%p difference**
- Confirmed liquidity-driven edge (prevent overfitting)

---

## Motivation

The initial goal was to develop an alpha strategy leveraging BTC 1-minute bar data. However, three critical issues were identified during validation:

**1. Data Noise**
- 1-min target: unpredictable, discarded
- 15-min target lag-1 autocorrelation: -0.05 (random noise level)
- 1-hour target lag-1 autocorrelation: 0.59 (weak but usable)
- **4-hour target lag-1 autocorrelation: 0.81 → adopted** (strong momentum signal)

**2. Feature Importance Issue**

The core liquidity features (SMC: BOS, CHoCH, Order Blocks, FVG) that were theoretically most critical for liquidity event detection. But, near-zero feature importance showed in validation. 

Binance API data quality issues(>80% NaN or noise) resulted in these features practically unusable. Despite being the theoretical backbone of the strategy, they were to be removed entirely.

Ironically, MDD improved significally after removal. Learned low-quality features introduce noise rather than signal, regardless of their theoretical releveance. 

**3. Insufficient Samples in Bull Markets**
- Liquidity events are rare during quiet bull markets
- Trade count dropped to single digits (2023: 9 trades, 2024: 1 trade)
- Statistical reliability could not be established for alpha generation

**Conclusion**

Data confirmed that the real edge lies not in generating alpha during bull markets, but in **defending against losses during bear markets and market stress events**.

The strategy was repositioned as a **risk management model**:
- Activates when liquidity events occur (forced liquidations, volatility explosions)
- Stays inactive during quiet bull markets

---

## Model Overview

This model detects **liquidity-driven events** in cryptocurrency futures markets and captures short-term mean reversion following price dislocations.

**Why liquidity events?**

The cryptocurrency futures market is highly leveraged. When prices move against leveraged positions, forced liquidations cascade. These dislocations revert quickly and create a tradeable edge.

**Why inactive in bull markets?**

Liquidity events are rare when markets trend smoothly upward. Low signal frequency in bull markets is not a model failure. It reflects the absence of the specific conditions where the model was designed to leverage.

---

## Methodology

### Data
- **Source**: Binance USDT-M Perpetual Futures
- **Assets**: BTCUSDT (primary), ETHUSDT (out of sample validation)
- **Timeframe**: 15-minute bars
- **Period**: 2019-09-25 ~ 2026-04-23 (~230,000 bars)
- **Features**: 100 features across momentum, volatility, volume, order flow, open interest, funding rate, long/short ratio, time

> Raw data was not included due to file size (~500MB). See `src/config.py` for data collection setup.

### Target Definition
4 hour direction label (16 bars × 15 min) with ±0.8% threshold:
- `+1`: price rises ≥ +0.8% after 4 hours
- `0`: price stays within ±0.8%
- `-1`: price falls ≥ -0.8% after 4 hours

**Why 4 hour horizon?**

| Target | Lag-1 Autocorrelation | Decision |
|--------|----------------------|----------|
| 1 min | Unpredictable | Discarded |
| 15 min ±0.08% | -0.05 | Pure noise |
| 1 hour ±0.3% | 0.59 | Weak |
| **4 hour ±0.8%** | **0.81** | **Adopted** |

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

- **Entry**: Long signal when predicted up probability > 0.40
- **Stop Loss**: Entry price − 2×ATR (dynamic, volatility-adjusted)
- **Max Hold**: 64 bars (16 hours)
- **Cooldown**: 16 bars (4 hours) after exit

**Why Long-only?**
Adding short signals marginally improved overall returns, but inroduced inconsistency across walkforward years. 
In addition, since the primary objective is risk management rather than return maximization, short signals were disabled. A model that occasionally generates higher returns but behaves unpredictably contradicts the core purpose of downside protection. 

### Walk-Forward Validation

```
Training: Rolling 2-year window 
Testing:  1-year in-sample, out-of-sample
Safety:   Skip if training data < 700 days
```

**Why rolling window?**
Cryptocurrency markets experience frequent distribution shifts. Older data introduces noise rather than signal. Rolling window ensures the model learns from recent market regimes only.

**Why 700 day minimum?**
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

**Note on SMC features**: Smart Money Concept features (BOS, CHoCH, Order Blocks, FVG) were initially included as core liquidity event detection. However, Binance API data quality issues (>80% NaN or noise) led to their removal. **MDD improved significantly after removal.**

### Seed Robustness

| Metric | Value |
|--------|-------|
| Seeds tested | 10 (0, 1, 2, 3, 7, 42, 100, 2024, 9999, 31415) |
| Positive CAGR | 10/10 |
| Average CAGR | 10.84% |
| CAGR Std | 3.90% |

Low standard deviation confirmed results were not dependent on a lucky random seed.

---

## Results

### BTC Strategy (5-Year Walk-Forward: 2021–2025)

| Year | Return | Sharpe | MDD | Trades | Win Rate | BnH |
|------|--------|--------|-----|--------|----------|-----|
| 2021 | 0.00% (blocked) | — | — | 0 | — | +64.2% |
| 2022 | **-11.69%** | N/A* | -22.10% | 64 | 0.438 | -64.2% |
| 2023 | +13.84% | N/A* | -2.13% | 9 | 0.667 | +156.1% |
| 2024 | +9.72% | N/A* | 0.00% | 1 | 1.000 | +118.2% |
| 2025 | **+23.97%** | 1.430 | -4.80% | 25 | 0.480 | -5.6% |

> *Sharpe marked N/A when trade count < 30 (statistically unreliable)
> 2026 excluded: partial year data (5 months only)

### ETH Out-of-Sample (Same Model, Same Parameters)

| Year | Return | MDD | Trades | BnH |
|------|--------|-----|--------|-----|
| 2021 | 0.00% (blocked) | — | 0 | +409.1% |
| 2022 | **+14.07%** | -47.83% | 228 | -67.6% |
| 2023 | +12.18% | 0.00% | 1 | +92.5% |
| 2024 | +23.08% | -6.46% | 11 | +46.3% |
| 2025 | **+9.36%** | -24.90% | 129 | -11.4% |

### Risk Management Validation

| Scenario | BTC BnH | BTC Strategy | ETH BnH | ETH Strategy |
|----------|---------|-------------|---------|-------------|
| 2022 Crash | -64.2% | **-11.69%** | -67.6% | **+14.07%** |
| 2025 Bear | -5.6% | **+23.97%** | -11.4% | **+9.36%** |

> Both assets. Same model. Same parameters. Consistent bear market defense.

---

## Alternative Approaches

Honest record of what was tried and why they were abandoned.

| Attempt | Result | Reason Abandoned |
|---------|--------|-----------------|
| 1-min / 15-min target | Failed | Lag-1 AC -0.05, pure noise |
| Long + Short | Failed | Short signals inconsistent across all years |
| HMM Regime Filter (3-state) | Failed | Distribution shift, misclassified BTC regimes |
| HMM Persistence Feature | Failed | Information overlap, degraded existing model |
| Dynamic Allocation (MA50/200) | Failed | Lagging signal, CAGR dropped to 8.3% |
| SMC Features | Removed | Binance API data quality >80% NaN/noise |
| ETH Portfolio Expansion | Abandoned | ETH strategy MDD -47.83% (2022), operationally unreliable |
| Expanding Window Walk-Forward | Rejected | Distribution shift makes old data noise in crypto |

---

## Limitations & Lessons Learned

### Data Issues

**1. SMC Feature Data Quality**
Smart Money Concept features (BOS, CHoCH, Order Blocks, FVG) were theoretically the most relevant indicators for liquidity event detection. However, Binance API data showed >80% NaN or noise. Removal improved MDD but learned the importance of data quality. 

**2. Short-Timeframe Noise**
1-min and 15-min targets confirmed pure noise (Autocorrelation: -0.05). Switched to 4-hour horizon (Autocorrelation: 0.81).

**3. ETH Data Start Date**
BTC data starts 2019-09-25, ETH starts 2019-11-27 (2-month gap). Walk-forward start date was adjusted. 

### Model Issues

**4. Regime Detection (HMM)**
Hidden Markov Model tested for regime classification. Distribution shift caused misclassification. Simple MA alignment (regime3) proved more stable. Sophisticated model was not outperformed. 

**5. Short Signal Inconsistency**
Short signals showed inconsistent win rates across all 6 walk-forward years. Long-only strategy was confirmed.

**6. Seed Dependency**
Single seed results are anecdotal. 10-seed validation confirmed CAGR std of 3.90%.

### Strategy Issues

**7. Insufficient Samples in Bull Markets**
a. 2023: 9 trades / 2024: 1 trade. Sharpe ratio was unreliable below 30 trades. 
b. Root cause: liquidity events rare during quiet bull markets.

**8. Dynamic Allocation Failure**
MA50/200 crossover tested for dynamic allocation. Lagging signal caused CAGR to drop to 8.3%. Static allocation proved superior.

**9. ETH Strategy MDD**
ETH strategy showed -47.83% MDD during 2022 despite +14.07% annual return. Operationally unreliable. ETH used only for out of sample validation.

### Validation Issues

**10. Walk-Forward Safety Guard**
2021 training data: only 463 days (< 700-day minimum). Trade was blocked and held only USD-Tether. Without this guard, model would generate unreliable signals.

---

## How to Run

### Packages
```
pandas
numpy
xgboost
pyarrow
scikit-learn
matplotlib
```

```bash
pip install -r requirements.txt
```

### Data Preparation
Raw data was not included due to file size constraints (~500MB).

```
1. Collect 15-min OHLCV + derivatives data from Binance USDT-M Futures
   Reference: src/config.py

2. Generate target labels:
   python src/make_targets_btc.py

3. Run BTC in-sample walk-forward backtest:
   python validator/walkforward1_BTC.py

4. Run ETH out-of-sample validation:
   python validator/walkforward2_ETH.py
```

### Pre-computed Results
Available in `results/`:
- `6y_walkforward_safe_results.csv` — BTC 6-year walk-forward
- `ETH_walkforward_results.csv` — ETH out-of-sample validation

---

## Repository Structure

```
systematic-bear-market-defense/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   ├── config.py              ← Data collection configuration
│   └── make_targets_btc.py    ← Target label generation
├── models/
│   └── feature_cols.json      ← 100 selected features
├── results/
│   ├── 6y_walkforward_safe_results.csv
│   └── ETH_walkforward_results.csv
└── validator/
    ├── walkforward1_BTC.py    ← BTC 6-year walk-forward
    └── walkforward2_ETH.py    ← ETH out-of-sample validation
```

---

## Future Work

**Project 2 — Statistical Mean Reversion (Alpha Generator)**
- Bollinger Band ±2σ breakout with several regime filters
- Multi-timeframe: A day or 4-hour signal & 1 hour entry
- CLT-based justification (window ≥ 30)
- Utilize Massive API Data
- Leverage MoE

**Project 3 — Option Pricing Model**
- Black-Scholes Model with option greeks in deep learning

**Improvement Areas**
- Higher quality liquidity data (order book, on-chain)
- SMC feature re-validation with better data source
- Improved regime detection in transitional markets
- Cross-exchange data integration
- Live trading validation

---

*Data source: Binance USDT-M Perpetual Futures*
*Backtest period: 2021–2025 (5-year walk-forward)*
*Out-of-sample: ETH validation with identical model setup (same model, same parameters)*
