import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import cvxpy as cp
import requests
from bs4 import BeautifulSoup
import seaborn as sns 
import matplotlib.pyplot as plt



# data functions
@st.cache_resource
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(
        zip(df["Symbol"], df["Security"])
    )
    return tickers, tickers_companies_dict

@st.cache_resource
def get_dax_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/DAX")
    df = df[4]
    tickers = df["Ticker"].to_list()
    tickers_companies_dict = dict(
        zip(df["Ticker"], df["Company"])
    )
    return tickers, tickers_companies_dict

@st.cache_resource
def get_nikkei_components():
    # Define the URL
    url = "https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/"

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the table based on its class attribute (you may need to inspect the HTML to get the exact class name)
        table = soup.find('table', {'class': 'tablepress'})

        # Use Pandas to read the table and store it as a DataFrame
        df = pd.read_html(str(table))[0]
        df['Code'] = df['Code'].astype(str) + '.T'
    else:
        print("Failed to retrieve the web page. Status code:", response.status_code)
    tickers = df["Code"].to_list()
    tickers_companies_dict = dict(
        zip(df["Code"], df['Company Name'])
    )
    return tickers, tickers_companies_dict

@st.cache_resource
def get_ftse_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/FTSE_100_Index")
    df = df[4]
    tickers = df["Ticker"].to_list()
    tickers_companies_dict = dict(
        zip(df["Ticker"], df["Company"])
    )
    return tickers, tickers_companies_dict


@st.cache_resource
def get_cac40_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/CAC_40")
    df = df[4]
    tickers = df["Ticker"].to_list()
    tickers_companies_dict = dict(
        zip(df["Ticker"], df["Company"])
    )
    return tickers, tickers_companies_dict


@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start, end)

@st.cache_resource
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")


def display_data_preview(title, dataframe, file_name="close_stock_prices.csv", key=0):
    data_exp = st.expander(title)
    available_cols = dataframe.columns.tolist()
    columns_to_show = data_exp.multiselect(
        "Columns", 
        available_cols, 
        default=available_cols,
        key=key
    )
    data_exp.dataframe(dataframe[columns_to_show])

    csv_file = convert_df_to_csv(dataframe[columns_to_show])
    data_exp.download_button(
        label="Download selected as CSV",
        data=csv_file,
        file_name=file_name,
        mime="text/csv",
    )


# Function to calculate annualized average returns and covariance matrix
def calculate_statistics(prices_df, n_days=252):
    returns_df = prices_df.pct_change().dropna()
    avg_returns = returns_df.mean() * n_days
    cov_mat = returns_df.cov() * n_days
    return avg_returns, cov_mat

# Function to generate unique markers based on the number of assets
def generate_markers(n_assets):
    marker_pool = ["o", "X", "d", "*", "^", "s"]  # Add more markers if needed
    return marker_pool[:n_assets]


def print_portfolio_summary(perf, weights, assets, name):
    """
    Helper function for printing the performance summary of a portfolio.

    Args:
        perf (pd.Series): Series containing the perf metrics
        weights (np.array): An array containing the portfolio weights
        assets (list): list of the asset names
        name (str): the name of the portfolio
    """
    name_portf = f"{name} portfolio Performance: ------------------"
    st.write(name_portf)
    for index, value in perf.items():
        st.write(f"{index}: {100 * value:.2f}% ", end="", flush=True)
    st.write("\nWeights")
    for x, y in zip(assets, weights):
        st.write(f"{x}: {100*y:.2f}% ", end="", flush=True)



# functions for calculating portfolio returns and volatility
def get_portf_rtn(w, avg_rtns):
    return np.sum(avg_rtns * w)

def get_portf_vol(w, avg_rtns, cov_mat):
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

# Function to calculate efficient frontier using SciPy optimization
def get_efficient_frontier_scipy(avg_returns, cov_mat, rtns_range):
        efficient_portfolios_scipy = []

        n_assets = len(avg_returns)
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = n_assets * [1. / n_assets, ]

        for ret in rtns_range:
            constr = (
                {"type": "eq",
                 "fun": lambda x: get_portf_rtn(x, avg_returns) - ret},
                {"type": "eq",
                 "fun": lambda x: np.sum(x) - 1}
            )
            ef_portf_scipy = sco.minimize(get_portf_vol,
                                          initial_guess,
                                          args=(avg_returns, cov_mat),
                                          method="SLSQP",
                                          constraints=constr,
                                          bounds=bounds)
            efficient_portfolios_scipy.append(ef_portf_scipy)

        return efficient_portfolios_scipy


def neg_sharpe_ratio(w, avg_rtns, cov_mat, rf_rate):
    portf_returns = np.sum(avg_rtns * w)
    portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    portf_sharpe_ratio = (
        (portf_returns - rf_rate) / portf_volatility
    )
    return -portf_sharpe_ratio
    