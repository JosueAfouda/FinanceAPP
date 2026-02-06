# FinanceAPP - GEMINI Project Context

## Project Overview
**FinanceAPP** is a comprehensive Python-based web application built with **Streamlit** for financial market analysis. It empowers users to perform technical analysis, forecast stock prices using machine learning (Prophet), and optimize portfolios using modern portfolio theory.

The app supports major global indices: **S&P500** (US), **CAC40** (France), **FTSE100** (UK), **DAX** (Germany), and **Nikkei 225** (Japan).

## Tech Stack & Architecture

### Core Technologies
*   **Language:** Python 3.12
*   **Framework:** Streamlit (Multipage App structure)
*   **Data Sources:** 
    *   `yfinance` (Market data)
    *   `requests` + `BeautifulSoup` + `pandas` (Scraping index components from Wikipedia/TopForeignStocks)

### libraries
*   **Visualization:** `plotly`, `cufflinks`, `matplotlib`, `seaborn`
*   **Machine Learning & Forecasting:** `prophet` (Facebook Prophet), `scikit-learn` (Metrics: MAE, RMSE, MAPE)
*   **Optimization:** `scipy.optimize`, `cvxpy` (Portfolio optimization)
*   **Utilities:** `pandas`, `numpy`, `joblib`, `lxml`

## Key Files & Directories

*   **`Home.py`**: The main entry point of the application. Handles the landing page, contact form, and sidebar navigation.
*   **`utils.py`**: Contains core utility functions:
    *   **Scrapers**: Robust functions (`get_sp500_components`, etc.) with User-Agent headers to fetch ticker symbols dynamically.
    *   **Data Loading**: `load_data` with `@st.cache_data` for performance.
    *   **Financial Math**: Functions for portfolio statistics (`calculate_statistics`), efficient frontier (`get_efficient_frontier_scipy`), and Sharpe ratio.
*   **`pages/`**: Contains the individual application modules:
    *   `01_TechnicalAnalysis.py`: Interactive charting and technical indicators.
    *   `02_Forecasting.py`: Prophet-based forecasting module. (Note: Recently refactored based on `AUDIT_FORECASTING.md` to include metrics, caching, and proper download logic).
    *   `03_AssetAllocation.py`: Portfolio optimization and Monte Carlo simulations.
    *   `04_Backtesting.py`: (Structure implies existence, likely for strategy backtesting).
*   **`AUDIT_FORECASTING.md`**: An audit report highlighting previous issues in the forecasting module (broken downloads, lack of metrics). *Analysis indicates these issues have been addressed in the current code.*
*   **`submissions.csv`**: Stores contact form submissions from the Home page.

## Building & Running

### Prerequisites
*   Python 3.12 (Required for specific dependency compatibility)
*   Virtual environment recommended.

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Execution
```bash
# Run the application
streamlit run Home.py
```

## Development Conventions

1.  **Caching:** The app heavily utilizes Streamlit's caching mechanisms to optimize performance:
    *   `@st.cache_data`: For data loading (e.g., `load_data`).
    *   `@st.cache_resource`: For heavy computations and model loading (e.g., `train_prophet_model`, scrapers).
2.  **Scraping Robustness:** Scrapers in `utils.py` use custom `User-Agent` headers to avoid 403 Forbidden errors.
3.  **Visualization:** Preference for interactive `plotly` charts over static images.
4.  **Forecasting Workflow:** 
    *   Train/Test split for evaluation (Metrics: MAE, RMSE, MAPE).
    *   Retraining on full data for future prediction.
    *   Dynamic holiday handling based on the selected market index.
5.  **Refactoring Pattern:** Evidence suggests a workflow of Auditing -> Refactoring (as seen with `AUDIT_FORECASTING.md`). Check for such documents before major changes.
