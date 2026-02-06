# Audit Report: Forecasting Module (`pages/02_Forecasting.py`)

## 1. Executive Summary
The "Forecasting" page aims to predict stock closing prices using the **Prophet** library. It allows users to select an asset, define a training/testing split, and visualize future price trends.

While the page functions as a basic prototype, it lacks critical components required for a robust production-grade forecasting tool. Key issues include the **absence of performance metrics** (users cannot objectively evaluate the model), **inefficient resource usage** (training the model twice), and **flawed download logic** that renders the "Download Model" feature unusable in cloud environments. From a financial perspective, forecasting raw stock prices without cross-validation or comparing against a baseline (like a random walk) can be misleading.

## 2. Critical Issues & Bugs

### 2.1. Broken Model Download Logic
**Issue:** The application saves the serialized model to the server's local filesystem (`serialized_model.json`) but does not provide a mechanism for the user to download this file to their local machine.
**Impact:** On cloud platforms (e.g., Streamlit Cloud, Render), the file is saved to an ephemeral container and is inaccessible to the user.
**Correction:** Use `st.download_button` with an in-memory byte stream (e.g., `io.BytesIO`) to allow direct client-side downloads.

### 2.2. Lack of Performance Metrics
**Issue:** The app visualizes "Actual vs. Predicted" graphs but does not calculate or display error metrics (e.g., **RMSE**, **MAE**, **MAPE**).
**Impact:** Users interpret results based solely on visual fit, which can be deceptive. There is no quantitative basis to judge if the model is accurate.
**Correction:** Calculate metrics on the test set and display them using `st.metric`.

### 2.3. Redundant Model Training
**Issue:** The model is initialized and trained **twice** in every run:
1.  Once on the training set for evaluation.
2.  Once on the full dataset for future forecasting.
**Impact:** This doubles the computation time and resource consumption unnecessarily.
**Correction:** While retraining on the full dataset is a valid strategy for final deployment, it should potentially be an optional step or optimized. At minimum, cache the results.

### 2.4. No Caching for Computationally Expensive Steps
**Issue:** The `prophet.fit()` method is not cached.
**Impact:** Any interaction that triggers a Streamlit rerun (e.g., changing a display option) forces the model to retrain from scratch, leading to a poor user experience.
**Correction:** Encapsulate the training logic in a function decorated with `@st.cache_resource`.

## 3. Modeling & Financial Logic Analysis

### 3.1. Evaluation Method (Split vs. CV)
**Current Approach:** A simple, sequential Train/Test split.
**Critique:** While standard, a single split is often insufficient for time series with changing volatility regimes.
**Recommendation:** Implement **Time Series Cross-Validation** (Rolling Window) using `prophet.diagnostics.cross_validation` to provide a more robust error estimate.

### 3.2. Raw Price Forecasting
**Critique:** Predicting raw closing prices ($P_t$) often results in a model that simply mimics the previous day's price ($P_{t-1}$), capturing the trend but failing to predict turning points.
**Recommendation:** Consider forecasting **Log Returns** or offering it as an advanced option. At the very least, warn users about the limitations of raw price trend extrapolation.

### 3.3. Holiday Mismatch
**Issue:** The user manually selects the `Country Holidays` (e.g., 'US', 'FR').
**Risk:** A user might select "CAC40" (French index) stocks but keep the default "US" holidays, confusing the model.
**Recommendation:** Automatically link the default holiday country to the selected Market Index (e.g., CAC40 -> FR, S&P500 -> US).

### 3.4. Frequency Assumption (`freq='B'`)
**Issue:** The code uses `freq='B'` (Business Days) for generating future dates.
**Risk:** This assumes markets are open every weekday, ignoring specific market holidays. This causes misalignment between the forecast index and actual trading days.
**Recommendation:** Use the specific trading calendar (if available via libraries like `pandas_market_calendars`) or rely on Prophet's default gap handling, though `freq='D'` with gaps is often safer if specific trading calendars aren't strictly enforced.

## 4. Code Quality & Best Practices

*   **Variable Naming:** The variable `btc_test` (Line 233) suggests code was copy-pasted from a Bitcoin-related tutorial. It should be renamed to something generic like `df_merged` or `stock_test`.
*   **Hardcoded Paths:** Saving files to the root directory is not thread-safe in multi-user environments.
*   **Visualization Logic:** The merge logic for plotting (`df_test.merge(df_pred...)`) is slightly brittle. Plotly can handle independent traces without merging dataframes, which would be cleaner.
*   **Deprecated Parameters:** The code contains `use_column_width` which was recently fixed in other files but should be ensured here as well (already addressed in previous steps, but worth noting for maintenance).

## 5. Recommendations for Refactoring

1.  **Refactor Training Logic:** Create a `train_prophet_model` function and cache it.
2.  **Add Metrics:** Implement a `calculate_metrics(y_true, y_pred)` function and display results in columns above the charts.
3.  **Fix Download:** Replace the file write operation with:
    ```python
    model_json = model_to_json(new_prophet)
    st.download_button("Download Model", data=model_json, file_name="model.json")
    ```
4.  **Auto-select Holidays:** Map `market_index` to a default country code for the `country_holidays` widget.
5.  **Clean Up Visualization:** Remove unused variables (`btc_test`) and simplify the Plotly trace generation.
