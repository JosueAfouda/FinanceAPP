# Audit Report: Asset Allocation Module

**Date:** February 6, 2026  
**Module:** `pages/03_AssetAllocation.py`  
**Auditor:** Gemini CLI (Senior Quantitative Finance Engineer)

---

## 1. Executive Summary

This report presents a technical and functional audit of the Asset Allocation module within the FinanceAPP application. The module implements three active portfolio optimization strategies (Monte Carlo, SciPy/SLSQP, CVXPY) and one placeholder for Hierarchical Risk Parity (HRP).

**Overall Assessment:**  
The implementation provides a solid educational and prototypical baseline for Modern Portfolio Theory (MPT). However, it suffers from **significant hardcoded limitations** (specifically in return ranges and risk-free rates) that severely restrict its robustness in real-world market scenarios. The **SciPy implementation is fragile**, while the **CVXPY implementation is mathematically more robust** but lacks clarity in variable naming.

---

## 2. Methodology-Specific Audit

### 2.1. Monte Carlo Simulations

**Financial Theory:**  
Approximates the Efficient Frontier by generating a large number of random portfolio weight vectors and calculating their resulting risk-return profiles.

**Implementation Analysis:**  
- **Logic:** Generates $10^5$ random portfolios using Dirichlet-like sampling (random uniform normalized to sum to 1).
- **Frontier Extraction:** Attempts to draw the "Efficient Frontier" line by binning the random portfolios and finding the minimum volatility for discrete return levels.

**Issues & Risks:**
- **Sampling Inefficiency:** For $N < 5$ assets, $10^5$ points cover the space well. For $N > 10$, the "curse of dimensionality" means the simulations will likely fail to sample the true optimal edge of the envelope, leading to a sub-optimal "Efficient Frontier" visualization.
- **Pseudo-Frontier:** The line drawn is not a mathematical efficient frontier; it is merely the "best of the random samples," which creates a misleading sense of optimality.

**Recommendations:**
1.  **Reduce Sample Size or Use Quasi-Random Numbers:** $10^5$ is computationally expensive for Python loops. Consider using Sobol sequences for better coverage with fewer points.
2.  **Clarify Visuals:** Label the line "Simulated Boundary" rather than "Efficient Frontier" to distinguish it from the analytical solution.

### 2.2. SciPy Optimization (Mean-Variance)

**Financial Theory:**  
Uses a non-linear solver (SLSQP) to numerically minimize portfolio volatility for a specific target return (constrained optimization).

**Implementation Analysis:**  
- **Objective:** Minimizes Volatility ($\sigma_p$) subject to $\sum w_i = 1$ and $\mu_p = 	ext{Target}$.
- **Frontier Construction:** Loops through a range of target returns to trace the curve.

**Critical Flaws:**
- **Hardcoded Return Range (CRITICAL):** The code uses `rtns_range = np.linspace(-0.1, 0.55, 200)`.
    - **Risk:** If the selected assets' actual annualized returns fall outside this range (e.g., a high-growth tech stock with 80% return or a crash scenario with -20%), the solver will **fail or produce a truncated frontier**. The frontier must be dynamic, bounded by `min(individual_asset_returns)` and `max(individual_asset_returns)`.
- **Hardcoded Risk-Free Rate:** `RF_RATE = 0` is hardcoded inside the `neg_sharpe_ratio` function calls.
    - **Risk:** This inflates the Sharpe Ratio in high-interest-rate environments (like 2023-2024), misleading the user about the risk-adjusted performance.

**Recommendations:**
1.  **Dynamic Bounds:** Calculate `rtns_range` dynamically based on the min and max historical returns of the selected assets.
2.  **User-Defined Risk-Free Rate:** Add a sidebar input for the Risk-Free Rate (defaulting to a reasonable proxy like 2.0% or 4.0%), or fetch the current 10Y Treasury yield.

### 2.3. CVXPY Optimization

**Financial Theory:**  
Reformulates the Mean-Variance problem as a convex optimization problem (Quadratic Programming), guaranteeing a global optimum if one exists.

**Implementation Analysis:**  
- **Objective:** Maximize $R_p - \gamma \cdot \sigma^2_p$. This traces the frontier by varying the risk-aversion parameter $\gamma$.

**Issues & Risks:**
- **Variable Naming Confusion:** The variable `portf_vol_cvx` calculates `cp.quad_form(weights, cov_mat)`, which is **Variance** ($\sigma^2$), not Volatility ($\sigma$). While the math remains correct for the optimizer, this naming is confusing for maintainers.
- **Scale Sensitivity:** The `gamma` range (`logspace(-3, 3)`) is heuristic. Depending on the scale of returns (decimals vs. percentage points), this range might miss parts of the frontier.

**Recommendations:**
1.  **Rename Variables:** Change `portf_vol_cvx` to `portf_var_cvx` to accurately reflect the mathematical object (Variance).
2.  **Robustness:** CVXPY is the preferred method over SciPy for this class of problems due to stability. Consider making this the default recommended method in the UI.

### 2.4. Hierarchical Risk Parity (HRP)

**Current Status:**  
Placeholder ("will be available soon").

**Assessment:**  
HRP is a modern technique (Lopez de Prado, 2016) that addresses the instability of the covariance matrix inversion in Mean-Variance optimization. It uses clustering (linkage) to group similar assets and allocates risk recursively.

**Recommendations:**
1.  **Library Usage:** Do not implement HRP from scratch. Use established libraries like `PyPortfolioOpt` which has robust HRP implementations.
2.  **Value Proposition:** When implemented, explain to the user that HRP usually yields more diversified portfolios than Mean-Variance, which often concentrates weights in a few assets.

---

## 3. Cross-Cutting Analysis

### 3.1. Data & Preprocessing
- **Adjusted Close:** Correctly uses `'Adj Close'` prices, which account for dividends and splits. This is crucial for total return accuracy.
- **Simple Returns:** Uses `pct_change()` (Simple Returns).
    - **Note:** For portfolio aggregation ($\sum w_i r_i$), simple returns are mathematically correct. Log returns should not be weighted-summed. The implementation is **correct**.
- **Missing Data:** `dropna()` is used. If assets have different IPO dates (starting at different times), this will truncate the entire dataset to the shortest history. **Risk:** Adding a recent IPO stock will discard years of data for other stable stocks.

### 3.2. Performance & Scalability
- **Repeated Calculations:** The `calculate_statistics` function is called, but the SciPy optimization loop (200 iterations) runs sequentially. For >10 assets, this will cause noticeable UI lag.
- **Redundant Training:** Similar to the Forecasting module, data fetching and basic stats could be cached more aggressively if the user switches between allocation methods without changing assets.

### 3.3. UX & Business Risks
- **Long-Only Constraint:** The bounds `(0, 1)` enforce a "No Short Selling" rule. This is appropriate for a general-purpose tool but should be explicitly stated in the UI so users don't expect hedge-fund-style strategies.
- **Interpretation of "Optimal":** Users may blindly follow the "Max Sharpe" weights. A disclaimer is needed stating that these weights are **backward-looking** (in-sample optimization) and do not guarantee future performance.

---

## 4. Action Plan & Recommendations

### Immediate Fixes (High Priority)
1.  **Dynamic Efficient Frontier Range (SciPy):** Replace the hardcoded `np.linspace(-0.1, 0.55, 200)` with dynamic bounds derived from `avg_returns.min()` and `avg_returns.max()`.
2.  **Parameterize Risk-Free Rate:** Pass `rf_rate` as an argument to `neg_sharpe_ratio` and allow the user to set it (or default to a non-zero value like 0.02).

### Architectural Improvements (Medium Priority)
1.  **Switch Default to CVXPY:** Promote CVXPY as the primary engine due to its superior numerical stability over SLSQP.
2.  **Refine Monte Carlo:** Reduce the number of simulations or optimize the vectorization. 100k points is excessive for a simple visualization.
3.  **Handling Short History:** Warn users if `dropna()` removes more than 50% of the data points (common when mixing old and new assets).

### Documentation & Safety
1.  **Rename Variable:** Fix the `portf_vol_cvx` (Variance) naming in the CVXPY block.
2.  **Disclaimer:** Add a clear "Past performance is not indicative of future results" note near the optimal weights table.
