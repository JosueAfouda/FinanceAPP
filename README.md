# FinanceAPP

## Overview
FinanceAPP is a comprehensive financial market trading web application developed using Streamlit. It empowers users to make informed investment decisions by providing advanced tools for:
*   **Technical Analysis**: Interactive charting with indicators.
*   **Forecasting**: Machine learning-based stock price prediction.
*   **Asset Allocation**: Portfolio optimization using modern portfolio theory.

The application supports major global indices: **S&P500** (USA), **CAC40** (France), **FTSE100** (UK), **DAX** (Germany), and **NIKKEI 225** (Japan).

## Key Features

### 1. Technical Analysis
*   **Interactive Charts**: Powered by `plotly` and `cufflinks`.
*   **Indicators**: 
    *   Simple Moving Average (SMA)
    *   Bollinger Bands
    *   Relative Strength Index (RSI)
    *   Volume Analysis

### 2. Forecasting (Prophet)
*   **Model**: Facebook's Prophet library for time-series forecasting.
*   **Smart Automation**: Automatically selects country holidays based on the chosen market index.
*   **Performance Metrics**: Evaluates models using **MAE**, **RMSE**, and **MAPE** on a test set.
*   **Visualization**: Interactive plots of historical data, future predictions, uncertainty intervals, and trend changepoints.
*   **Downloadable Models**: Export trained models as JSON files for external use.

### 3. Asset Allocation
*   **Optimization Techniques**:
    *   **Monte Carlo Simulations**: Randomly generates thousands of portfolios to visualize the risk-return trade-off.
    *   **SciPy Optimization**: Mathematically finds the Efficient Frontier and optimal portfolios (Max Sharpe, Min Volatility).
    *   **CVXPY Optimization**: Convex optimization for robust and efficient portfolio construction.
    *   **Hierarchical Risk Parity (HRP)**: (Coming Soon)

## Installation

### Prerequisites
*   **Python 3.12** (Strictly required for compatibility with specific financial libraries).
*   **Pip** package manager.

### Setup Steps
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/FinanceAPP.git
    cd FinanceAPP
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit Application:**
    ```bash
    streamlit run Home.py
    ```

2.  **Navigate the App:**
    *   **Home**: Landing page with general information and a contact form.
    *   **Technical Analysis**: Select an index and ticker to view charts and indicators.
    *   **Forecasting**: Train a Prophet model to predict future prices and download the results.
    *   **Asset Allocation**: Select multiple assets to build and optimize a portfolio.

## Technical Details

*   **Data Sources**: Real-time stock data via `yfinance`. Index components are scraped dynamically from Wikipedia using `requests` and `pandas` with custom headers for robustness.
*   **Caching**: Extensive use of `@st.cache_resource` and `@st.cache_data` ensures fast performance by caching data and trained models.
*   **Deployment**: Ready for deployment on platforms like Render (includes `.python-version` and `lxml` support).

## Project Structure
*   `Home.py`: Main entry point.
*   `utils.py`: Core utility functions for data fetching (robust scraping), processing, and financial calculations.
*   `pages/`:
    *   `01_TechnicalAnalysis.py`: Interactive charting module.
    *   `02_Forecasting.py`: Advanced forecasting module with caching and metrics.
    *   `03_AssetAllocation.py`: Portfolio optimization module.
*   `requirements.txt`: Pinned dependencies for reproducible builds.

## Author
Developed by **Josué AFOUDA**.
Connect with me on [YouTube](https://www.youtube.com/@RealProDatascience).

## License
This project is licensed under the MIT License.
