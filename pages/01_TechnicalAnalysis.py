# imports
import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot
import requests
from bs4 import BeautifulSoup

from utils import *

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
    datetime.date(2022, 1, 1)
)
end_date = st.sidebar.date_input(
    "End date", 
    datetime.date.today()
)

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

## inputs for technical analysis
st.sidebar.header("Technical Analysis Parameters")

volume_flag = st.sidebar.checkbox(label="Add volume")

exp_sma = st.sidebar.expander("SMA")
sma_flag = exp_sma.checkbox(label="Add SMA")
sma_periods= exp_sma.number_input(
    label="SMA Periods", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)

exp_bb = st.sidebar.expander("Bollinger Bands")
bb_flag = exp_bb.checkbox(label="Add Bollinger Bands")
bb_periods= exp_bb.number_input(label="BB Periods", 
                                min_value=1, max_value=50, 
                                value=20, step=1)
bb_std= exp_bb.number_input(label="# of standard deviations", 
                            min_value=1, max_value=4, 
                            value=2, step=1)

exp_rsi = st.sidebar.expander("Relative Strength Index")
rsi_flag = exp_rsi.checkbox(label="Add RSI")
rsi_periods= exp_rsi.number_input(
    label="RSI Periods", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)
rsi_upper= exp_rsi.number_input(label="RSI Upper", 
                                min_value=50, 
                                max_value=90, value=70, 
                                step=1)
rsi_lower= exp_rsi.number_input(label="RSI Lower", 
                                min_value=10, 
                                max_value=50, value=30, 
                                step=1)

# main body

st.title("Technical Analysis")

run_button = st.sidebar.button("Run Analysis")

if run_button:

    df = load_data(ticker, start_date, end_date)

    ## data preview part
    display_data_preview("Preview data", df, file_name=f"{ticker}_stock_prices.csv", key=1)

    ## technical analysis plot
    title_str = f"{tickers_companies_dict[ticker]}'s stock price"
    qf = cf.QuantFig(df, title=title_str)
    if volume_flag:
        qf.add_volume()
    if sma_flag:
        qf.add_sma(periods=sma_periods)
    if bb_flag:
        qf.add_bollinger_bands(periods=bb_periods,
                            boll_std=bb_std)
    if rsi_flag:
        qf.add_rsi(periods=rsi_periods,
                rsi_upper=rsi_upper,
                rsi_lower=rsi_lower,
                showbands=True)

    fig = qf.iplot(asFigure=True)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, height=500)