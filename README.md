# BTC/USDT Intraday Volatility Forecasting & Adaptive Trend Strategy

**IITG.ai Summer Research Project · 2026**

> *Deep learning volatility forecasting + systematic trend-following on Bitcoin.*
> *+33.24% net return vs −35.79% buy-and-hold in the Jan–Jun 2026 bear market.*

---

## Results at a Glance

| Metric | Result |
|---|---|
| **Phase 1: QLIKE improvement** | **+56.7%** over HAR-RV baseline |
| Best forecasting model | Jump-BiLSTM + Attention |
| Best model QLIKE | 0.4319 vs baseline 0.9900 |
| Calibrated band coverage | 89.9% (target: 90%) |
| News sentiment impact | <0.5% — null finding |
| **Phase 2: Strategy Sharpe** | **1.838** vs B&H −1.935 |
| Net return (test period) | **+33.24%** vs −35.79% |
| Max drawdown | −20.68% vs −42.21% |
| Trades / avg hold | 16 trades / ~11 days |
| Total fees on $10k | ~$50 |

---

## Project Structure

```
📁 notebooks/
   z2-all-models-comparison.ipynb    ← Phase 1: model training & evaluation
   z4-adaptive-trend-btc-vol.ipynb    ← Phase 2: strategy backtesting

📁 presentation/
   Presentation.pdf
```

---

## The Two Phases

### Phase 1 — Volatility Forecasting

**Goal:** Predict 30-minute-ahead realised volatility from 5-minute BTC/USDT bars.

**Data:** 368,333 bars · Jan 2023–Jun 2026 · KuCoin via CCXT
**Split:** Train Jan 2023–Jun 2025 · Val Jul–Dec 2025 · Test Jan–Jun 2026 (run once)

**21 input features** including:
- HAR-RV lags (Corsi, 2009)
- Jump decomposition: RV_continuous, RV_pos_jump, RV_neg_jump (Barndorff-Nielsen et al.)
- ETH cross-asset volatility (Zhang et al., 2024)
- Deribit DVOL implied vol index
- Intraday seasonality encoding

**Three models benchmarked:**

| Model | QLIKE | vs Baseline |
|---|---|---|
| HAR-RV (Corsi 2009) | 0.9900 | — baseline |
| HAR-LSTM | 0.4689 | +52.6% |
| DeepVol TCN | 0.4661 | +53.1% |
| **Jump-BiLSTM + Attention** | **0.4319** | **+56.7%** ← best |

**Key findings:**
- All three DL architectures converge to QLIKE 0.43–0.47 → the **feature set is the information ceiling**, not model capacity
- BiLSTM wins via bidirectional processing + attention over 78 timesteps + explicit jump features
- FinBERT news sentiment: <0.5% QLIKE change → **clean null result** (99.7% of bars have zero news)
- Calibrated bands: k=2.15 → 89.9% coverage on test (target 90%)

---

### Phase 2 — Trading Strategy

**Goal:** Exploit BiLSTM vol predictions in a fee-efficient trading strategy.

**Two strategies that failed (and why):**

| Strategy | Result | Root Cause |
|---|---|---|
| VWAP Mean Reversion | 0/6 months positive (walk-forward) | Ranging val → trending test: regime mismatch |
| Moreira-Muir Vol-Scaling | Underperformed buy-and-hold | Vol-return relationship is **inverted** in crypto |

**AdaptiveTrend (arXiv:2602.11708) — what worked:**

Trend-following on 6H bars (aggregated from 5-min):

```
Entry:  EMA(20) crossover → LONG above, SHORT below. Always in market.
Exit:   Trailing stop — only ratchets in profit direction, never widens.

stop_distance = k × ATR(14) × vol_factor
vol_factor    = BiLSTM_pred_vol / median_train_pred_vol

High pred_vol → wider stop → ride volatile rallies
Low pred_vol  → tighter stop → lock gains faster
```

**One parameter:** k=4.0 selected on VAL, locked before TEST. Zero leakage.

**Test results — Jan–Jun 2026 (BTC fell $109k → $73k):**

| Metric | AdaptiveTrend | Buy & Hold |
|---|---|---|
| Sharpe (bar-level) | **1.838** | −1.935 |
| Total return | **+33.24%** | −35.79% |
| Max drawdown | −20.68% | −42.21% |
| Monthly wins vs B&H | 5/6 | — |

**Statistical note:** Bootstrap 95% CI = [−1.109, +4.639]. Economically significant, not statistically proven at 95% given 16 trades over 6 months. Walk-forward over 24–36 months would be the standard for confirmation.

---

## Data & Model Weights

Large files (parquet, model weights, predictions) are hosted on Kaggle — not included in this repo (GitHub 100MB limit).

| File | Location |
|---|---|
| `merged_5min.parquet` (OHLCV + features) | Kaggle dataset |
| `bilstm_5min_best.pt` (model weights) | Kaggle dataset |
| `predictions_5min.npz` (pred_vol arrays) | Kaggle dataset |

---

## Tech Stack

```
Python 3.10  ·  PyTorch  ·  NumPy  ·  Pandas  ·  scikit-learn
CCXT (KuCoin API)  ·  Kaggle T4 GPU  ·  Matplotlib
```

---

## References

1. Andersen et al. (2003) — Realized volatility from high-frequency returns
2. Corsi (2009) — HAR-RV model
3. Barndorff-Nielsen & Shephard (2004) — Bipower variation / jump decomposition
4. Zhang et al. (2024, JFEC) — BiLSTM + ETH cross-asset features
5. Moreno-Pino & Zohren (2024) — DeepVol TCN
6. Moreira & Muir (2017, JF) — Volatility-managed portfolios (tested, failed on crypto)
7. Duc Bui (arXiv:2602.11708, 2026) — AdaptiveTrend

---

*IITG.ai Summer Research Programme 2026*
