# FinanceAPP - Gemini Context

## Project Overview
**FinanceAPP** is a web-based financial analysis tool built with **Streamlit**. It empowers users to analyze stock market data from major global indices (S&P500, CAC40, FTSE100, DAX, NIKKEI 225) through technical analysis, forecasting, and asset allocation strategies.

## Key Features
*   **Technical Analysis:** Interactive charts with Simple Moving Average (SMA), Bollinger Bands, and RSI using `cufflinks` and `plotly`.
*   **Forecasting:** Stock price prediction using Facebook's `prophet`.
*   **Asset Allocation:** Portfolio optimization (Efficient Frontier, Sharpe Ratio) using `cvxpy` and `scipy`.
*   **Multi-Index Support:** Dynamically fetches components for various global indices via web scraping (`beautifulsoup4`, `pandas`).

## Development & Usage

### Prerequisites
*   Python 3.x
*   Pip

### Installation
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
To start the Streamlit server:
```bash
streamlit run Home.py
```
The application typically runs on `http://localhost:8501`.

### Data Handling
*   **Stock Data:** Fetched in real-time using `yfinance`.
*   **Index Components:** Scraped from Wikipedia and other sources using `requests` with a custom `User-Agent` header to prevent `HTTPError` (cached via `@st.cache_resource` in `utils.py`).
*   **User Submissions:** Contact form submissions are saved locally to `submissions.csv`.

## Codebase Structure

### Core Files
*   **`Home.py`**: The main entry point. Sets up the landing page, navigation, and contact form.
*   **`utils.py`**: Contains critical utility functions:
    *   Data fetching (`get_sp500_components`, `load_data`, etc.).
    *   Financial calculations (`calculate_statistics`, `get_efficient_frontier_scipy`).
    *   Visualization helpers.

### Pages (`pages/`)
Streamlit uses this directory to automatically create a multi-page app structure:
*   **`01_TechnicalAnalysis.py`**: Interactively plots stock data with user-selected technical indicators.
*   **`02_Forecasting.py`**: Implements Prophet for time-series forecasting.
*   **`03_AssetAllocation.py`**: Performs portfolio optimization and visualization.
*   **`04_Backtesting.py`**: (Currently a placeholder) Intended for strategy backtesting.

### Configuration
*   **`requirements.txt`**: Lists all Python dependencies.
*   **`.devcontainer/`**: Configuration for VS Code Dev Containers.
*   **`submissions.csv`**: Local storage for user messages.

## Development Conventions
*   **UI Framework:** Exclusively uses Streamlit.
*   **Data Science Stack:** Relies heavily on `pandas` for data manipulation, `numpy` for numerical operations, and `plotly`/`cufflinks` for interactive plotting.
*   **Caching:** Uses Streamlit's `@st.cache_resource` and `@st.cache_data` decorators to optimize performance, especially for web scraping and data fetching functions in `utils.py`.
