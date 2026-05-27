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

The initial goal was to develop an alpha strategy using BTC 15-minute bar data. However, two critical issues were identified during validation: 

**1. Data Noise**
  - 15-min target lag-1 autocorrelation: -0.05 (random noise level)
  - Unpredictable at short horizons
  - Switched to 4 hour targets: lag-1 autocorrelation 0.81(learnable momentum signal)

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

This model detects **liquidityy-driven events** in cryptocurrency futures markets and captures short-term mean reversion by following price dislocations.















