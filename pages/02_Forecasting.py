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
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_plotly
import plotly.express as px
from prophet.serialize import model_to_json

from utils import *

st.title("Forecasting Close Price")

#st.subheader('Technical Analysis Page')
# df = "ma dataframe"
#from Home import my_var

#st.write(my_var)

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

# available_tickers, tickers_companies_dict = get_sp500_components()

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
country_holidays = exp_prophet.selectbox("Country Holidays", ['US', 'FR', 'DE', 'JP', 'GB'])
horizon = exp_prophet.number_input("Forecast Horizon (days)", min_value=1, value=365, step=1)
download_prophet = exp_prophet.checkbox(label="Download Model")

# Parameters for Auto Arima
exp_arima = st.sidebar.expander("Auto Arima Parameters")
test_data_percentage2 = exp_arima.number_input("Percentage of Test Data", 0.1, 0.4, 0.2, 0.05)
m = exp_arima.number_input("The period for seasonal differencing", min_value=1, value=12, step=1)
information_criterion = exp_arima.selectbox("Information Criterion", ['aic', 'bic', 'hqic', 'oob'])
test_stat = exp_arima.selectbox("Type of stationarity test", ['adf', 'kpss'])
seasonal = exp_arima.checkbox(label="Whether to fit a seasonal ARIMA")
intercept = exp_arima.checkbox(label="Whether to include an intercept term")
method = exp_arima.selectbox("Which solver is used", ['lbfgs', 'nm', 'bfgs', 'powell', 'cg'])
max_iter = exp_arima.number_input("Maximum number of iterations", min_value=10, value=50, step=1)


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

        # Modeling process with Prophet
        st.write("Running Prophet Modeling Process...")
        # ... Add your Prophet modeling code here ...
        df = df[['Close']]
        df = df.reset_index(drop=False)
        df.columns = ['ds', 'y']

        # Sequential train/test split with 80% for training and 20% for testing
        df_train, df_test = train_test_split(df, 
                                             test_size=test_data_percentage, 
                                             shuffle=False, 
                                             random_state=42)

        # Create and fit the model
        prophet = Prophet(changepoint_range=changepoint_range)
        prophet.add_country_holidays(country_name=country_holidays)
        #prophet.add_seasonality(name="annual", period=365, fourier_order=5)
        prophet.fit(df_train)

        # Predictions on test data
        df_future = prophet.make_future_dataframe(
            periods=len(df_test),
            freq="B" # Business Days
        )
        df_pred = prophet.predict(df_future)

        ## Prediction preview part
        display_data_preview("Prediction Data", df_pred, file_name=f"{ticker}_pred_data.csv", key=3)

        # Plot the results
        fig = plot_plotly(prophet, df_pred)
        st.plotly_chart(fig)

        # Changepoints

        # Create a dataframe with the changepoints
        changepoints = pd.DataFrame(prophet.changepoints)
        display_data_preview("Changepoints Data", 
                             changepoints, 
                             file_name=f"{ticker}_changepoints.csv",
                             key=4)

        # Create a Plotly figure
        fig = go.Figure()

        # Add a line for the actual data
        fig.add_trace(go.Scatter(
            x=df_pred['ds'],
            y=df_pred['yhat'],
            mode='lines',
            name='Actual Close Price'
        ))

        # Add scatter points for changepoints
        fig.add_trace(go.Scatter(
            x=changepoints['ds'],
            y=df_pred[df_pred['ds'].isin(changepoints['ds'])]['yhat'],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
            ),
            name='Changepoints'
        ))

        # Update the layout for better visualization
        fig.update_layout(
            title=f"{ticker} Price - Actual vs. Changepoints",
            xaxis_title="Date",
            yaxis_title=f"{ticker} Price ($)",
        )

        # Show the interactive Plotly chart
        st.plotly_chart(fig)

        
        # Affichage des prix vs predictions dans un même graphique
        # merge the test values with the forecasts
        SELECTED_COLS = [
            "ds", "yhat", "yhat_lower", "yhat_upper"
        ]

        df_pred = (
            df_pred
            .loc[:, SELECTED_COLS]
            .reset_index(drop=True)
        )
        df_test = df_test.merge(df_pred, on=["ds"], how="left")
        df_test["ds"] = pd.to_datetime(df_test["ds"])
        btc_test = df_test.set_index("ds")

        # Create a figure for the interactive chart
        fig = go.Figure()

        # Plot the actual values ('y')
        fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['y'], mode='lines', name='Actual', line=dict(color='blue')))

        # Plot the predicted values ('yhat')
        fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['yhat'], mode='lines', name='Predicted', line=dict(color='orange')))

        # Fill the region between the lower and upper bounds
        fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, fill='tonexty', fillcolor='rgba(255,165,0,0.3)', name='Uncertainty'))

        fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,165,0,0.3)', showlegend=False))

        # Customize the layout
        fig.update_layout(
            title=f"{tickers_companies_dict[ticker]}'s Close Price - Actual vs. Predicted",
            xaxis_title="Date",
            yaxis_title=f"{tickers_companies_dict[ticker]}'s Close Price ($)",
        )

        # Show the interactive chart
        st.plotly_chart(fig)

        ############################################## Pevisions (in futur dates) ######################
        # Create and fit the model
        new_prophet = Prophet(
            changepoint_range=changepoint_range
        )
        new_prophet.add_country_holidays(country_name=country_holidays)
        #prophet.add_seasonality(name="annual", 
                                #period=365, 
                                #fourier_order=5)
        new_prophet.fit(df)

        # Forecasts (in the future)
        future = new_prophet.make_future_dataframe(
            periods=horizon,
            freq="B" # Business Days
        )
        forecasts = new_prophet.predict(future)

        last_date = df['ds'].max()
        forecasts = forecasts[forecasts['ds'] > last_date]

        ## Prediction preview part
        display_data_preview("Forecasts Data", forecasts, file_name=f"{ticker}_forecasts_data.csv", key=5)

        # Create a Plotly figure to display the actual and forecasted prices
        fig = go.Figure()

        # Plot the actual prices
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Price'))

        # Plot the forecasted prices
        fig.add_trace(go.Scatter(x=forecasts['ds'], y=forecasts['yhat'], mode='lines', name='Forecasted Price', line=dict(color='red')))

        # Customize the layout
        fig.update_layout(
            title=f"{ticker} Price - Actual vs. Forecasted",
            xaxis_title="Date",
            yaxis_title=f"{ticker} Price",
        )

        # Display the interactive chart using Streamlit
        st.plotly_chart(fig)


        # Download the Model
        if download_prophet:
            with open('serialized_model.json', 'w') as fout:
                fout.write(model_to_json(new_prophet))
            st.success("Prophet Model downloaded successfully as 'serialized_prophet_model.json'")