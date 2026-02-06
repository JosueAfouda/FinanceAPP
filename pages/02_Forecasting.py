# imports
import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_plotly
import plotly.express as px
from prophet.serialize import model_to_json

from utils import *

# Helper functions for Forecasting
@st.cache_resource
def train_prophet_model(data, changepoint_range, country_holidays):
    """
    Trains a Prophet model. Cached to avoid redundant computation.
    """
    model = Prophet(changepoint_range=changepoint_range)
    if country_holidays:
        model.add_country_holidays(country_name=country_holidays)
    model.fit(data)
    return model

def calculate_metrics(y_true, y_pred):
    """
    Calculates MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, rmse, mape

st.set_page_config(page_title="Financial Market Trading web app", layout="wide")
# Conteneur pour aligner les éléments horizontalement
col1, col2, col3 = st.columns([1, 4, 1])

# Colonne gauche : Image
with col1:
    st.image(
        "linkedin_profil.png",  # Remplacez par le chemin de votre image
        width=80,     # Ajustez la taille si nécessaire
        use_container_width=False,
    )

# Colonne centrale : Titre
with col2:
    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 0;'>Financial Market Trading web app</h1>
        """,
        unsafe_allow_html=True,
    )

# Colonne droite : Nom et lien LinkedIn
with col3:
    st.markdown(
        """
        <div style='text-align: right;'>
            <a href="https://www.linkedin.com/in/josu%C3%A9-afouda/" target="_blank" style='text-decoration: none; color: #0077b5;'>
                <strong>Josué AFOUDA</strong>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.title("Forecasting Close Price")

## set offline mode for cufflinks
cf.go_offline()

# sidebar

## inputs for downloading data
st.sidebar.header("Stock Parameters")

# Update available tickers based on market index selection
market_index = st.sidebar.selectbox(
    "Market Index", 
    ["S&P500", "CAC40", "DAX", "FTSE100", "Nikkei225"]
)

# Map market index to default holiday country
index_holiday_map = {
    "S&P500": "US",
    "CAC40": "FR",
    "DAX": "DE",
    "FTSE100": "GB",
    "Nikkei225": "JP"
}
default_holiday = index_holiday_map.get(market_index, "US")

if market_index == "S&P500":
    available_tickers, tickers_companies_dict = get_sp500_components()
elif market_index == "CAC40":
    available_tickers, tickers_companies_dict = get_cac40_components()
elif market_index == "DAX":
    available_tickers, tickers_companies_dict = get_dax_components()
elif market_index == "FTSE100":
    available_tickers, tickers_companies_dict = get_ftse_components()
elif market_index == "Nikkei225":
    available_tickers, tickers_companies_dict = get_nikkei_components()

ticker = st.sidebar.selectbox(
    "Ticker", 
    available_tickers, 
    format_func=tickers_companies_dict.get
)
start_date = st.sidebar.date_input(
    "Start date", 
    datetime.date(2019, 1, 1)
)
end_date = st.sidebar.date_input(
    "End date", 
    datetime.date.today()
)

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

## inputs for technical analysis
st.sidebar.header("Forecasting Process")

exp_prophet = st.sidebar.expander("Prophet Parameters")
test_data_percentage = exp_prophet.number_input("Testing Data Percentage", 0.1, 0.4, 0.2, 0.05)
changepoint_range = exp_prophet.number_input("Changepoint Range", 0.05, 0.95, 0.9, 0.05)

# Set default index based on the mapped default_holiday
holiday_options = ['US', 'FR', 'DE', 'JP', 'GB']
try:
    default_index = holiday_options.index(default_holiday)
except ValueError:
    default_index = 0

country_holidays = exp_prophet.selectbox("Country Holidays", holiday_options, index=default_index)
horizon = exp_prophet.number_input("Forecast Horizon (days)", min_value=1, value=365, step=1)

#st.subheader("Modeling Process")
modeling_option = st.sidebar.radio("Select Modeling Process", ["Prophet"])


# main body

run_button = st.sidebar.button("Run Forecasting")

if run_button:

    df = load_data(ticker, start_date, end_date)
    #df.dropna(inplace=True)

    ## data preview part
    display_data_preview("Preview data", df, key=2)

    ## plot close price
    close_plot = st.expander("Close Price Chart")
    # Plot the close price data
    fig = go.Figure()
    title_str = f"{tickers_companies_dict[ticker]}'s Close Price"
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=title_str,
                      xaxis_title='Date',
                      yaxis_title='Close Price ($)')
    st.plotly_chart(fig)

    if modeling_option == "Prophet":

        st.write("### 1. Model Evaluation (Train/Test Split)")
        st.write("Running Prophet Modeling Process on Training Data...")
        
        # Prepare data for Prophet
        df_prophet = df[['Close']].reset_index()
        df_prophet.columns = ['ds', 'y']

        # Sequential train/test split
        df_train, df_test = train_test_split(df_prophet, 
                                             test_size=test_data_percentage, 
                                             shuffle=False, 
                                             random_state=42)

        # Train model on Training set (Cached)
        model_train = train_prophet_model(df_train, changepoint_range, country_holidays)

        # Predictions on test data
        future_test = model_train.make_future_dataframe(periods=len(df_test), freq="B")
        # We only need predictions for the test period
        forecast_test = model_train.predict(future_test)
        
        # Filter forecasts to match test set dates for metric calculation
        # Note: 'ds' in forecast_test might include weekends if freq='D', but here we use 'B'.
        # Safest way is to merge on 'ds'.
        df_merged = df_test.merge(forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')

        # Calculate Metrics
        mae, rmse, mape = calculate_metrics(df_merged['y'], df_merged['yhat'])
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:.2f}")
        m2.metric("RMSE", f"{rmse:.2f}")
        m3.metric("MAPE", f"{mape:.2%}")

        # Plot Actual vs Predicted (Test Set)
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=df_merged['ds'], y=df_merged['y'], mode='lines', name='Actual', line=dict(color='blue')))
        fig_eval.add_trace(go.Scatter(x=df_merged['ds'], y=df_merged['yhat'], mode='lines', name='Predicted', line=dict(color='orange')))
        fig_eval.add_trace(go.Scatter(x=df_merged['ds'], y=df_merged['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, fill='tonexty', fillcolor='rgba(255,165,0,0.3)', name='Uncertainty'))
        fig_eval.add_trace(go.Scatter(x=df_merged['ds'], y=df_merged['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,165,0,0.3)', showlegend=False))
        
        fig_eval.update_layout(title="Model Evaluation: Actual vs Predicted (Test Set)", xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig_eval)


        st.write("### 2. Future Forecast (Full Data)")
        st.write("Retraining model on full dataset for future prediction...")

        # Train model on Full dataset (Cached)
        model_full = train_prophet_model(df_prophet, changepoint_range, country_holidays)

        # Forecasts (in the future)
        future_full = model_full.make_future_dataframe(periods=horizon, freq="B")
        forecast_full = model_full.predict(future_full)

        # Filter for only future dates
        last_date = df_prophet['ds'].max()
        forecast_future = forecast_full[forecast_full['ds'] > last_date]

        ## Prediction preview part
        display_data_preview("Forecasts Data (Future)", forecast_future, file_name=f"{ticker}_forecasts_data.csv", key=5)

        # Plot Actual + Forecast
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Historical Price'))
        fig_future.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat'], mode='lines', name='Future Forecast', line=dict(color='red')))
        fig_future.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, fill='tonexty', fillcolor='rgba(255,0,0,0.2)'))
        fig_future.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,0,0,0.2)', showlegend=False))

        fig_future.update_layout(title=f"Future Forecast ({horizon} days)", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_future)

        # Changepoints Analysis (on full model)
        st.write("### 3. Trend Analysis")
        changepoints = pd.DataFrame(model_full.changepoints)
        
        fig_cp = go.Figure()
        fig_cp.add_trace(go.Scatter(x=forecast_full['ds'], y=forecast_full['yhat'], mode='lines', name='Trend'))
        fig_cp.add_trace(go.Scatter(
            x=changepoints['ds'],
            y=forecast_full[forecast_full['ds'].isin(changepoints['ds'])]['yhat'],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Changepoints'
        ))
        fig_cp.update_layout(title="Trend Changepoints", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_cp)

        # Correct Download Logic
        model_json = model_to_json(model_full)
        st.download_button(
            label="Download Trained Model (JSON)",
            data=model_json,
            file_name=f"{ticker}_prophet_model.json",
            mime="application/json"
        )

else:
    st.write("""
    ### User Manual: Forecasting Menu

    ---

    #### **Welcome to the Forecasting Menu**

    This section of the application enables you to forecast the closing price of stocks using advanced machine learning techniques. Follow the instructions below to configure your parameters and generate predictions.

    ---

    ### **How to Use the Forecasting Menu**

    #### **Step 1: Configure Stock Parameters**
    1. **Select a Market Index**:
    - Choose a market index (e.g., S&P500, CAC40, DAX, FTSE100, Nikkei225) from the dropdown menu in the sidebar.
    - The tickers of the selected index will populate the dropdown menu for easy selection.

    2. **Choose a Ticker**:
    - Select a company ticker from the dropdown menu. The corresponding company name will be displayed.

    3. **Set the Date Range**:
    - Select the **start date** and **end date** for the data to be analyzed.
    - Ensure the start date precedes the end date; otherwise, you will be prompted to adjust the dates.

    ---

    #### **Step 2: Define Forecasting Parameters**
    1. **Prophet Parameters** (Expand the "Prophet Parameters" section in the sidebar):
    - **Testing Data Percentage**: Specify the proportion of data to use for testing (default: 20%).
    - **Changepoint Range**: Adjust the sensitivity of changepoint detection (default: 90%).
    - **Country Holidays**: Select the country for adding holiday effects to the model.
    - **Forecast Horizon (days)**: Set the number of days to forecast into the future (default: 365 days).
    
    2. **Modeling Process**:
    - Select "Prophet" from the radio buttons to enable Prophet-based modeling.

    ---

    #### **Step 3: Run Forecasting**
    1. Click the **"Run Forecasting"** button in the sidebar.
    2. The application will:
    - Retrieve the stock data for the selected ticker and date range.
    - Display a preview of the data.
    - Plot the historical closing prices.
    - Evaluate the model on a test set and display error metrics (MAE, RMSE, MAPE).
    - Generate future forecasts using the full dataset.

    ---

    ### **Output Visualizations and Data**
    1. **Model Evaluation**:
    - Visual comparison of Actual vs Predicted prices on the test set.
    - Key performance metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

    2. **Future Forecast**:
    - Interactive chart showing historical data and future predictions with uncertainty intervals.
    - Downloadable CSV of forecasted values.

    3. **Trend Analysis**:
    - Visualization of significant trend changepoints identified by the model.

    4. **Download Model**:
    - Download the fully trained Prophet model as a JSON file for future use (compatible with cloud environments).

    ---

    ### **Note for First-Time Users**
    - This manual will remain visible until you click "Run Forecasting."
    - Carefully configure all parameters to ensure accurate forecasting results.

    ---

    Enjoy forecasting with this tool! If you have any questions or encounter issues, double-check your inputs or seek assistance. 🚀
    """)