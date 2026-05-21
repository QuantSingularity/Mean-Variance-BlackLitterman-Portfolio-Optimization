# Mean-Variance-BlackLitterman-Portfolio-Optimization

> Quantitative Portfolio Management

> Full quantitative portfolio construction pipeline for 10 US equities: mean-variance optimization with and without weight constraints, Fama-French five-factor attribution, Monte Carlo simulation vs hard optimization convergence analysis, and Black-Litterman model with synthetic composite market portfolio.

---

## Overview

This project implements a complete institutional portfolio construction workflow across four methodological pillars. Each part builds on the previous, moving from raw return estimation through constrained optimization, factor attribution, simulation benchmarking, and finally Bayesian view incorporation via the Black-Litterman framework.

| Part   | Methodology                   | Key Output                                                                                     |
| ------ | ----------------------------- | ---------------------------------------------------------------------------------------------- |
| Part 1 | Mean-Variance Optimization    | Tangency portfolios with and without 18% weight cap, dual efficient frontiers, OOS performance |
| Part 2 | Fama-French Five-Factor Model | OLS and Robust Huber M-estimator factor attribution, exposure decomposition                    |
| Part 3 | Simulation vs Optimization    | Convergence analysis - simulations required for weights to match hard optimization             |
| Part 4 | Black-Litterman Model         | Synthetic composite market portfolio, posterior expected returns, view-adjusted allocation     |

---

## Universe and Data

**Assets:** TSLA, WMT, BAC, GS, LLY, MRK, GOOG, META, AAPL, XOM (10 S&P 500 constituents)

**Estimation period:** September 1, 2023 - September 30, 2025 (daily returns, ~521 observations)

**Out-of-sample period:** October 1, 2025 - December 31, 2025 (63 trading days)

**Market benchmark:** SPY (SPDR S&P 500 ETF), used for Black-Litterman market equilibrium construction

**Risk-free rate:** 5.00% per annum

**Data source:** Live prices via `yfinance` (`auto_adjust=True`). Fully self-contained synthetic GBM fallback included - no external files required.

---

## Part 1 - Mean-Variance Optimization

### Step 1 - Tangency Portfolio (No Short-Selling)

The tangency portfolio maximises the Sharpe ratio subject to non-negative weights summing to one. Solved using SLSQP with 30 random starting points to ensure the global optimum is found rather than a local maximum:

```
max  (w'mu - Rf) / sqrt(w'Sigma w)   s.t.  sum(w) = 1,  w >= 0
```

**Key finding:** The unconstrained tangency portfolio concentrates in only 2 of 10 assets (WMT: 27.99%, META: 72.01%), illustrating the well-known sensitivity of MV optimization to estimated mean returns.

### Step 2 - Constrained Portfolio (Max 18% Per Asset)

The same optimisation with an additional box constraint `w_i <= 0.18`. The cap forces diversification across 6 assets and demonstrates the efficiency cost of weight constraints:

```
max  (w'mu - Rf) / sqrt(w'Sigma w)   s.t.  sum(w) = 1,  0 <= w_i <= 0.18
```

### Step 3 - Efficient Frontier Comparison

Both efficient frontiers are computed by minimising portfolio variance at each target return level (120 grid points). The constrained frontier lies strictly inside the unconstrained frontier at every risk level.

### Step 4 - Out-of-Sample Performance (Oct-Dec 2025)

Both portfolios are evaluated over the 63-day out-of-sample period using:

- Cumulative return and annualised return
- Annualised Sharpe ratio
- Maximum drawdown and Calmar ratio

---

## Part 2 - Fama-French Five-Factor Model

The Step 2 portfolio (max 18% weight constraint) is regressed on the Fama-French five factors using daily excess returns. Data sourced from Kenneth French's data library via pandas_datareader, with direct URL download and synthetic fallback.

### Factor Definitions

| Factor | Name                          | Role                                                                   |
| ------ | ----------------------------- | ---------------------------------------------------------------------- |
| Mkt-RF | Market Premium                | Systematic market risk; beta measures equity market co-movement        |
| SMB    | Small Minus Big               | Size premium; negative loading expected for mega-cap portfolio         |
| HML    | High Minus Low                | Value premium; negative loading expected for growth-tilted portfolio   |
| RMW    | Robust Minus Weak             | Profitability premium; positive loading expected for high-margin firms |
| CMA    | Conservative Minus Aggressive | Investment premium; negative loading for R&D-intensive firms           |

### Regression Setup

- **Training:** 80% of aligned observations (chronologically first)
- **Testing:** remaining 20% - out-of-sample R-squared validation
- **OLS:** `statsmodels.OLS` with White heteroskedasticity-robust standard errors
- **Robust:** Huber M-estimator (`statsmodels.RLM`) to downweight extreme return days

---

## Part 3 - Simulation vs Hard Optimization

Five assets are selected from the ten: **AAPL, LLY, META, WMT, XOM** (spanning technology, healthcare, consumer staples, and energy).

### Step 7 - No Short-Selling

Dirichlet distribution samples are used (automatically satisfying sum=1 and w>=0). Convergence is measured by the L1 distance between simulated and exact optimal weights, tested at N = 500, 1k, 5k, 10k, 50k, 100k.

### Step 8 - 30% Maximum Weight Constraint

Rejection sampling applied to Dirichlet draws: samples where any weight exceeds 30% are discarded. Tested at N = 1k, 5k, 10k, 50k, 100k, 500k.

**Key finding:** Hard optimization (SLSQP) finds the exact global optimum in milliseconds. Simulation is approximate, slow to converge for sparse optima, and requires rejection sampling overhead for box constraints. Simulation cannot replace hard optimization for portfolio construction.

---

## Part 4 - Black-Litterman Model

### Step 9a - Synthetic Composite Market Portfolio

The market portfolio is built using the synthetic composite approach: 10 stocks assigned their approximate S&P 500 market-cap weights (totalling ~22.9%), with an 11th "rest of S&P 500" asset making up the remainder (~77.1%). The 11th asset return is computed as:

```
R_11 = (R_SPY - sum_i w_i * R_i) / w_11
```

The full 11-asset covariance matrix is used to derive the market equilibrium implied returns.

### Step 9b - Tau Parameter

```
tau = 1/T
```

where T is the number of training observations. This makes prior uncertainty equal to the statistical estimation uncertainty of the sample mean - the most theoretically rigorous choice for daily data with large T.

### Step 9c - Investor Views

Three views are expressed on the 10 tradable assets:

| View | Type     | Statement                             |
| ---- | -------- | ------------------------------------- |
| 1    | Relative | AAPL outperforms TSLA by +5% per year |
| 2    | Absolute | LLY achieves +20% annual return       |
| 3    | Relative | GOOG outperforms XOM by +3% per year  |

The view uncertainty matrix Omega is set using the He-Litterman/Idzorek method:

```
Omega = diag(P * (tau * Sigma) * P')
```

### Step 9d - BL Posterior and Optimal Weights

BL posterior formulas:

```
Sigma_post = inv(inv(tau*Sigma) + P' * inv(Omega) * P)
mu_BL      = Rf + Sigma_post @ (inv(tau*Sigma)*pi + P' * inv(Omega) * q)
Sigma_BL   = Sigma + Sigma_post
```

where `pi = delta * Sigma * w_mktcap` are the implied equilibrium excess returns and `delta = (mu_mkt - Rf) / sigma_mkt^2` is the market risk aversion coefficient.

Both unconstrained and 18%-capped BL portfolios are optimised using SLSQP with the BL parameters `(mu_BL, Sigma_BL)`.

### Step 9e - BL vs Part 1 Comparison

Discussion of how each view shifts portfolio weights, whether constraints amplify or dampen the BL effect, and how BL's market-equilibrium prior compares to the raw historical-mean approach used in Part 1.

---

## Tech Stack

```
Python 3.x
yfinance
scipy (optimize - SLSQP)
statsmodels (OLS, RLM)
pandas_datareader (Fama-French factors)
numpy
pandas
matplotlib
seaborn
```

---

## Installation

```bash
git clone https://github.com/QuantSingularity/Mean-Variance-BlackLitterman-Portfolio-Optimization.git
cd Mean-Variance-BlackLitterman-Portfolio-Optimization
pip install yfinance scipy statsmodels pandas_datareader numpy pandas matplotlib seaborn
jupyter notebook Portfolio-Optimization-MV-BL.ipynb
```

> Live price data and Fama-French factors are fetched automatically. If unavailable, fully self-contained synthetic datasets are generated - no external files required.

---

## Topics

`mean-variance-optimization` `black-litterman` `portfolio-optimization` `efficient-frontier` `fama-french` `factor-model` `monte-carlo` `simulation` `slsqp` `tangency-portfolio` `sharpe-ratio` `spy` `yfinance` `statsmodels` `quantitative-finance` `python` `QuantSingularity`

---

## References

- Black, F., & Litterman, R. (1992). Global Portfolio Optimization. _Financial Analysts Journal_, 48(5), 28-43.
- He, G., & Litterman, R. (2002). The Intuition Behind Black-Litterman Model Portfolios. Goldman Sachs Investment Management Division.
- Markowitz, H. (1952). Portfolio Selection. _Journal of Finance_, 7(1), 77-91.
- Fama, E. F., & French, K. R. (2015). A Five-Factor Asset Pricing Model. _Journal of Financial Economics_, 116(1), 1-22.
- Idzorek, T. (2005). A Step-by-Step Guide to the Black-Litterman Model. Zephyr Associates Working Paper.
- Lopez de Prado, M. (2018). _Advances in Financial Machine Learning_. Wiley.
