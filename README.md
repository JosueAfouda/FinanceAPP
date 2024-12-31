# FinanceAPP

## Overview
FinanceAPP is a comprehensive financial market trading web application developed using Streamlit. It provides users with tools for technical analysis, asset allocation, forecasting, and backtesting of financial data. The application is designed to help users make informed investment decisions by analyzing stock data from major global indices such as S&P500, CAC40, FTSE100, DAX, and NIKKEI 225.

## Features
- **Technical Analysis**: Analyze stock performance using indicators like SMA, Bollinger Bands, and RSI.
- **Asset Allocation**: Optimize investment portfolios using techniques like Monte Carlo simulations and CVXPY optimization.
- **Forecasting**: Predict future stock prices using the Prophet model.
- **Backtesting**: (Coming Soon) Evaluate trading strategies against historical data.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FinanceAPP.git
   ```
2. Navigate to the project directory:
   ```bash
   cd FinanceAPP
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the application using Streamlit:
```bash
streamlit run Home.py
```

## Project Structure
- **Home.py**: Main entry point for the application, handles user interface setup.
- **utils.py**: Contains utility functions for data retrieval and processing.
- **pages/**: Contains individual modules for different functionalities:
  - `01_TechnicalAnalysis.py`: Module for technical analysis.
  - `02_Forecasting.py`: Module for forecasting stock prices.
  - `03_AssetAllocation.py`: Module for asset allocation strategies.
  - `04_Backtesting.py`: Placeholder for future backtesting functionality.

## Author
Developed by Josué AFOUDA. Connect with me on [LinkedIn](https://www.linkedin.com/in/josu%C3%A9-afouda/).

## License
This project is licensed under the MIT License.