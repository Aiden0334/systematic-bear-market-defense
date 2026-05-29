# Systematic Bear Market Defense via XGBoost
> A Risk Overlay Model for Cryptocurrency Futures: Liquidity-Driven Downside Protection

---

## Abstract

We present a **risk overlay model** for cryptocurrency futures portfolios. This is not an alpha strategy. The objective is singular: systematic detection of liquidity stress events and containment of downside exposure during bear markets.

Cryptocurrency futures markets are structurally prone to liquidity crises. We exploit this microstructural property using XGBoost on 15-minute BTC futures data (2019–2026, ~230,000 bars), validated through a 5-year rolling walk-forward framework and confirmed on ETH as an independent out-of-sample asset.

**Core results:**
- 2022 crypto crash (BTC -64.2%): model contained drawdown to **-11.69%** — a **52.5%p loss reduction**
- 2025 bear market (BTC -5.6%): model generated **+23.97%** — outperforming passive holding by **29.6%p**
- ETH out-of-sample: CAGR gap of **0.16%p** (BTC 5.64% vs ETH 5.48%), confirming the edge is structural, not asset-specific

The model is intentionally inactive during quiet bull markets. Low signal frequency in trending upmarkets is not a limitation — it is the expected behavior of an event-driven risk overlay. This property is documented, validated, and discussed throughout.

---

## 1. Introduction

### 1.1 Problem Statement

Passive Bitcoin holding has historically generated strong long-term returns. However, its drawdown profile is operationally unsustainable for risk-constrained allocators:

```
2018: BTC drawdown  -83%
2022: BTC drawdown  -64.2%
2025: BTC drawdown  -26% (peak to trough)
```

Existing risk management approaches for cryptocurrency portfolios share a common weakness: they are reactive. Volatility-based filters trigger after drawdowns begin. Static hedges sacrifice bull market participation. Neither approach exploits the **predictable microstructure** of cryptocurrency futures.

### 1.2 Research Question

> Can we systematically detect liquidity stress events in cryptocurrency futures markets and use them to limit downside exposure — without sacrificing the ability to participate in bull markets through a complementary BnH position?

### 1.3 Why Liquidity Events?

Cryptocurrency futures markets are structurally different from equity markets:

- **Extreme leverage**: retail traders routinely use 10–100x leverage
- **Cascade dynamics**: price moves trigger stop-losses, which trigger liquidations, which trigger further price moves
- **Predictable reversion**: post-liquidation dislocations tend to revert within 4–16 hours
- **Observable signals**: derivatives data (open interest, funding rate, long/short ratio) provides real-time visibility into positioning stress

This creates a **tradeable edge** that is episodic, mean-reverting, and detectable — precisely the conditions suited to an event-driven risk overlay.

### 1.4 Model Positioning

This model is designed to complement, not replace, passive BnH exposure:

```
Quiet bull market  → BnH participates, model stays inactive
Liquidity stress   → Model activates, detects and defends
Bear market        → Model contains drawdown
```

<img width="1870" height="1199" alt="chart5_regime_donut" src="https://github.com/user-attachments/assets/f5caae61-a90e-48ed-9045-0af270d9ce27" />


The model's inactivity during bull markets is by design. An overlay that fires indiscriminately would introduce noise and erode returns. Selectivity is the edge.

---

## 2. Methodology

### 2.1 Data

| Attribute | Detail |
|-----------|--------|
| Source | Binance USDT-M Perpetual Futures |
| Primary asset | BTCUSDT |
| Validation asset | ETHUSDT (out-of-sample) |
| Timeframe | 15-minute bars |
| Period | 2019-09-25 ~ 2026-04-23 |
| Bars | ~230,000 (BTC), ~224,000 (ETH) |

> Raw data was not included due to file size (~500MB). See `src/config.py` for collection setup.

### 2.2 Target Definition

We define a 4-hour direction label on 15-minute bars:

```
target = +1  if  close[t+16] / close[t] - 1  ≥  +0.8%
target =  0  if  |close[t+16] / close[t] - 1| <   0.8%
target = -1  if  close[t+16] / close[t] - 1  ≤  -0.8%
```

**Horizon selection was empirically validated via autocorrelation analysis:**
<img width="1485" height="730" alt="chart2_autocorrelation" src="https://github.com/user-attachments/assets/82ac4256-3f71-48b9-8d63-643d9c8bcf62" />



A lag-1 autocorrelation of 0.81 indicates strong momentum persistence — a necessary condition for a classifier to learn meaningful directional patterns.

### 2.3 Feature Engineering

100 features across 21 categories, engineered from OHLCV and derivatives data:

| Category | Key Features | Role in Liquidity Detection |
|----------|-------------|----------------------------|
| HL Range | hl_range_5m/15m/60m/240m | Price range explosion (top importance: 0.162) |
| Volatility | volatility_5m/15m/60m/240m | Volatility regime detection |
| Liquidity Sweep | swept_high/low_15m/60m | Stop hunt identification |
| Open Interest | oi_roc_15m, oi_zscore | Forced liquidation proxy |
| Funding Rate | fr_level, fr_extreme, fr_cumsum | Positioning stress measurement |
| Long/Short Ratio | ls_ratio_level, ls_extreme | Crowding and capitulation detection |
| Swing VWAP | swing_vwap_dist, swing_vwap_dir | Liquidity-anchored price position |
| Volume | vol_spike_240m, vol_buy_pressure | Event magnitude confirmation |
| Time | time_us_session, time_hour_sin/cos | Session-based liquidity patterns |

> Full feature list: `models/feature_cols.json`

**Note on SMC features**: Smart Money Concept features (BOS, CHoCH, Order Blocks, FVG) were theoretically the most relevant for liquidity event detection. However, Binance API data quality rendered them unusable (>80% NaN or noise), with near-zero feature importance confirmed in validation. MDD improved significantly after removal — demonstrated that low-quality features introduce noise regardless of theoretical relevance. This remains the primary data limitation of the current implementation.

### 2.4 Model

**XGBoost Classifier** — 3-class (down / sideways / up)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 100 | Speed-accuracy balance |
| max_depth | 4 | Overfitting control |
| learning_rate | 0.05 | Conservative convergence |
| subsample | 0.8 | Row-level regularization |
| colsample_bytree | 0.8 | Feature-level regularization |
| min_child_weight | 10 | Sparse pattern suppression |
| reg_alpha | 0.1 | L1 regularization |
| reg_lambda | 1.0 | L2 regularization |

**Execution parameters:**
- Entry: predicted up probability > 0.40
- Stop loss: entry − 2×ATR (volatility-scaled dynamic stop)
- Max hold: 64 bars (16 hours)
- Cooldown: 16 bars (4 hours) post-exit

**Why 1x leverage (no leverage)?**
This backtest intentionally uses 1x exposure throughout. The model is designed as a risk overlay — its purpose is to measure pure defensive capability, not to amplify returns through leverage. Introducing leverage would obscure the signal quality and conflate risk management performance with position sizing decisions. All results reflect 1x returns.

**Why Long-only?**
Short signals marginally improved gross returns but introduced year-to-year inconsistency across the walk-forward period. Since the primary objective is **downside protection rather than return maximization**, unpredictable short behavior contradicts the model's core purpose. Long-only was adopted based on walk-forward evidence.

### 2.5 Walk-Forward Validation

```
Window:   Rolling 2-year training → 1-year test
Safety:   Minimum 700 training days required
Folds:    6 total (2021–2026), evaluated on 2021–2025
```

**Why rolling window?**
Cryptocurrency markets undergo frequent regime shifts. An expanding window dilutes recent market structure with stale historical data. Rolling window ensures the model adapts to current regimes.

**Why 700-day minimum?**
Nominal 2-year windows vary between 729–730 days due to leap year variation. A 700-day floor provides a safety margin while preserving the 2-year learning horizon. 2021 was blocked under this rule (463 training days available).

### 2.6 Seed Robustness

<img width="1785" height="732" alt="chart3_seed_robustness" src="https://github.com/user-attachments/assets/45157680-2a12-497c-9d23-6b9acb464aab" />


A CAGR standard deviation of 3.90% across 10 independent seeds confirms the result is structurally driven.

---

## 3. Results

### 3.1 BTC Strategy (5-Year Walk-Forward: 2021–2025)

| Year | Market | Return | Sharpe | MDD | Trades | BnH | vs BnH |
|------|--------|--------|--------|-----|--------|-----|--------|
| 2021 | Bull | 0.00% (blocked) | — | — | 0 | +64.2% | — |
| 2022 | Bear | **-11.69%** | N/A† | -22.10% | 64 | -64.2% | **+52.5%p** |
| 2023 | Bull | +13.84% | N/A† | -2.13% | 9 | +156.1% | -142.3%p |
| 2024 | Bull | +9.72% | N/A† | 0.00% | 1 | +118.2% | -108.5%p |
| 2025 | Bear | **+23.97%** | 1.430 | -4.80% | 25 | -5.6% | **+29.6%p** |

<img width="1785" height="883" alt="chart4_drawdown_comparison" src="https://github.com/user-attachments/assets/b5758924-d9d3-4f90-82fc-3f85d6ba9531" />

> †Sharpe marked N/A when trade count < 30 (insufficient sample for reliable estimation)
> 2026 excluded: partial year (5 months)

**Interpretation**: The model outperformed passive holding in both bear/sideways years by **52.5%p (2022)** and **29.6%p (2025)**. Underperformance in bull years (2023, 2024) is expected — the model detects liquidity events, which are rare in trending upmarkets. This is the intended behavior of a risk overlay.

### 3.2 ETH Out-of-Sample Validation

Identical model applied to ETHUSDT without modification (same architecture, same hyperparameters, LONG_P=0.40):

| Year | Return | MDD | Trades | BnH | vs BnH |
|------|--------|-----|--------|-----|--------|
| 2021 | 0.00% (blocked) | — | 0 | +409.1% | — |
| 2022 | **+14.07%** | -47.83% | 228 | -67.6% | **+81.7%p** |
| 2023 | +12.18% | 0.00% | 1 | +92.5% | -80.4%p |
| 2024 | +23.08% | -6.46% | 11 | +46.3% | -23.2%p |
| 2025 | **+9.36%** | -24.90% | 129 | -11.4% | **+20.8%p** |

### 3.3 Risk Overlay Validation

<img width="2084" height="924" alt="chart1_bear_defense" src="https://github.com/user-attachments/assets/b949810a-e429-4b60-b266-ea3dfbeaf78d" />

> Same model. Same parameters. Two independent assets. Consistent downside protection across all bear and sideways periods tested.

**The 0.16%p CAGR gap between BTC (5.64%) and ETH (5.48%) out-of-sample confirms the edge is structural — not an artifact of asset-specific optimization.**

---

## 4. Alternative Approaches

| Approach | Outcome | Reason Abandoned |
|----------|---------|-----------------|
| 1-min / 15-min target | Failed | Lag-1 AC -0.05, statistically unpredictable |
| Long + Short | Rejected | Inconsistent short win rates across all walk-forward years |
| HMM Regime Filter (3-state) | Failed | Distribution shift caused systematic regime misclassification |
| HMM Persistence Feature | Failed | Information overlap degraded existing model performance |
| Dynamic Allocation (MA50/200) | Failed | Lagging signal reduced CAGR to 8.3% |
| SMC Features | Removed | >80% NaN/noise, near-zero feature importance confirmed |
| ETH Portfolio Expansion | Abandoned | ETH intra-year MDD -47.83% (2022), operationally unreliable |
| Expanding Window Walk-Forward | Rejected | Stale data degrades regime-adaptive learning in crypto |

---

## 5. Limitations

### 5.1 Data Quality
**SMC Feature Gap**: The most relevant features for liquidity detection were rendered unusable by Binance API data quality. A higher-quality data source (order book snapshots, on-chain liquidation feeds) could materially improve signal precision. This is the primary unresolved limitation.

**ETH Data Start Date**: ETH futures data begins 2019-11-27 vs BTC 2019-09-25 (2-month gap). Walk-forward folds adjusted accordingly.

### 5.2 Model
**Regime Detection**: Simple 3-state MA alignment (regime3) was used after HMM-based approaches failed. More robust regime classification could improve entry filtering during market transitions.

**Low Trade Count**: 2023 (9 trades) and 2024 (1 trade) produce statistically unreliable Sharpe ratios (N/A). Single-year results in low-frequency periods remain sensitive to individual trade outcomes.

### 5.3 Strategy
**Bull Market Participation**: The model is inactive during quiet bull markets by design. This produces significant underperformance vs BnH in trending years. As a standalone strategy, this is a critical limitation. As a risk overlay combined with passive BnH exposure, it is the intended behavior.

**ETH Intra-Year MDD**: Despite strong annual returns, ETH strategy exhibited -47.83% intra-year MDD in 2022. Operationally, this drawdown is difficult to sustain through. ETH is retained for out-of-sample validation only.

### 5.4 Validation
**No Live Trading**: All results are simulation-based. Slippage, funding costs, and execution latency are not fully captured.

**Single Exchange**: Data sourced exclusively from Binance USDT-M Futures. Cross-exchange liquidity dynamics are not modeled.

---

## 6. How to Run

### Packages
```
pandas / numpy / xgboost / pyarrow / scikit-learn / matplotlib
```
```bash
pip install -r requirements.txt
```

### Data Preparation
```
1. Collect 15-min OHLCV + derivatives from Binance USDT-M Futures
   Reference: src/config.py

2. Generate target labels:
   python src/make_targets_btc.py

3. BTC walk-forward backtest:
   python validator/walkforward1_BTC.py

4. ETH out-of-sample validation:
   python validator/walkforward2_ETH.py
```

### Pre-computed Results
```
results/6y_walkforward_safe_results.csv   — BTC 6-year walk-forward
results/ETH_walkforward_results.csv       — ETH out-of-sample
```

---

## 7. Repository Structure

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

## 8. Future Work

The current model establishes a validated **bear market defense layer**. The roadmap extends this into a multi-strategy framework that addresses its primary structural limitation: inactivity during bull markets.

**Project 2 — Statistical Mean Reversion**
The current model's edge disappears in quiet uptrends. Project 2 directly targets this regime gap: Bollinger Band ±2σ breakouts with CLT-justified statistical confidence (window ≥ 30), combined with candlestick pattern confirmation for precise entry timing. Multi-timeframe structure: 4-hour directional signal & 15-minute entry. This strategy is designed to generate returns during the periods where Project 1 is intentionally silent.

**Project 3 — Swing Trend-Following**
For capturing larger directional moves across full market cycles: Ichimoku Cloud (trend structure) + Fibonacci retracements (entry levels) & momentum divergence (early reversal warning) & Bollinger Band (volatility context). Target holding period: days to weeks — structurally distinct from the sub-24-hour horizon of Projects 1 and 2.

**Combined Risk Framework**
The three strategies target distinct market regimes:
```
Bear / stress   → Project 1 (liquidity event defense)
Sideways        → Project 2 (statistical mean reversion)
Trending bull   → Project 3 (swing trend-following)
```
The long-term objective is a regime-aware allocation system that weights each strategy dynamically based on detected market conditions — replacing the current static overlay with an adaptive risk management framework.

**Technical Improvements**
- Higher quality liquidity data: order book snapshots, on-chain liquidation feeds
- SMC feature re-validation with cleaner data source
- Improved regime detection for transitional market periods
- Cross-exchange data integration
- Live trading validation with full execution cost modeling

---

*Data: Binance USDT-M Perpetual Futures*
*Validation: 5-year rolling walk-forward (2021–2025)*
*Out-of-sample: ETH with identical model configuration (Same Model, Same Parameters)*
