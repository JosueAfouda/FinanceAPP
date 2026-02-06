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
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

# Headers for requests to avoid bot detection
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


# HRP Functions
def get_cluster_var(cov, c_items):
    """
    Calculates the variance of a cluster.
    """
    cov_slice = cov.loc[c_items, c_items]
    ivp = 1. / np.diag(cov_slice)
    ivp /= ivp.sum()
    return np.dot(np.dot(ivp.T, cov_slice), ivp)

def get_rec_bisection(cov, sort_ix):
    """
    Performs recursive bisection to allocate weights.
    """
    w = pd.Series(1, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
    return w

def get_quasi_diag(link):
    """
    Sorts the clustered items to be quasi-diagonal.
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def get_hrp_weights(prices_df):
    """
    Calculates weights using Hierarchical Risk Parity (HRP).
    """
    returns = prices_df.pct_change().dropna()
    cov = returns.cov()
    corr = returns.corr()
    
    # 1. Tree Clustering
    # Calculate distance metric based on correlation
    dist = np.sqrt(0.5 * (1 - corr))
    link = sch.linkage(squareform(dist), 'single')
    
    # 2. Quasi-Diagonalization
    sort_ix_indices = get_quasi_diag(link)
    sort_ix = corr.index[sort_ix_indices].tolist()
    
    # 3. Recursive Bisection
    weights = get_rec_bisection(cov, sort_ix)
    return weights


# data functions
@st.cache_resource
def get_sp500_components():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=HEADERS)
    dfs = pd.read_html(response.text, flavor='lxml')
    
    # Search for the table with 'Symbol' and 'Security'
    df = None
    for d in dfs:
        if 'Symbol' in d.columns and 'Security' in d.columns:
            df = d
            break
    if df is None:
         # Fallback to index 0 if search fails, though less robust
         df = dfs[0]

    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(
        zip(df["Symbol"], df["Security"])
    )
    return tickers, tickers_companies_dict

@st.cache_resource
def get_dax_components():
    url = "https://en.wikipedia.org/wiki/DAX"
    response = requests.get(url, headers=HEADERS)
    dfs = pd.read_html(response.text, flavor='lxml')
    
    # Search for the table with 'Ticker' and 'Company'
    df = None
    for d in dfs:
        if 'Ticker' in d.columns and 'Company' in d.columns:
            df = d
            break
    if df is None:
        # Fallback to index 4 (legacy behavior) or 3 (common alternative)
        df = dfs[4] if len(dfs) > 4 else dfs[3]

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
    response = requests.get(url, headers=HEADERS)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the table based on its class attribute (you may need to inspect the HTML to get the exact class name)
        table = soup.find('table', {'class': 'tablepress'})

        # Use Pandas to read the table and store it as a DataFrame
        df = pd.read_html(str(table), flavor='lxml')[0]
        # Only append .T if not already present
        df['Code'] = df['Code'].astype(str).apply(lambda x: x if x.endswith('.T') else x + '.T')
    else:
        print("Failed to retrieve the web page. Status code:", response.status_code)
    tickers = df["Code"].to_list()
    tickers_companies_dict = dict(
        zip(df["Code"], df['Company Name'])
    )
    return tickers, tickers_companies_dict

@st.cache_resource
def get_ftse_components():
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    response = requests.get(url, headers=HEADERS)
    dfs = pd.read_html(response.text, flavor='lxml')
    
    # Search for the table with 'Ticker' or 'EPIC'
    df = None
    ticker_col = "Ticker"
    for d in dfs:
        if "Ticker" in d.columns:
            df = d
            ticker_col = "Ticker"
            break
        elif "EPIC" in d.columns:
            df = d
            ticker_col = "EPIC"
            break
            
    if df is None:
        # Fallback
        df = dfs[4]
    
    # Rename for consistency if found as EPIC
    if ticker_col != "Ticker":
        df = df.rename(columns={ticker_col: "Ticker"})
        
    # FTSE 100 tickers on Yahoo Finance usually need .L suffix
    df["Ticker"] = df["Ticker"].astype(str).apply(lambda x: x if x.endswith('.L') else x + '.L')

    tickers = df["Ticker"].to_list()
    tickers_companies_dict = dict(
        zip(df["Ticker"], df["Company"])
    )
    return tickers, tickers_companies_dict


@st.cache_resource
def get_cac40_components():
    url = "https://en.wikipedia.org/wiki/CAC_40"
    response = requests.get(url, headers=HEADERS)
    dfs = pd.read_html(response.text, flavor='lxml')
    
    # Search for the table with 'Ticker' and 'Company'
    df = None
    for d in dfs:
        if 'Ticker' in d.columns and 'Company' in d.columns:
            df = d
            break
    if df is None:
         df = dfs[4] if len(dfs) > 4 else dfs[3]
         
    tickers = df["Ticker"].to_list()
    tickers_companies_dict = dict(
        zip(df["Ticker"], df["Company"])
    )
    return tickers, tickers_companies_dict


@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start, end, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten if only one ticker is present to maintain compatibility with 
        # Technical Analysis and Forecasting pages.
        if len(df.columns.get_level_values(1).unique()) == 1:
            df.columns = df.columns.get_level_values(0)
    return df

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
    st.write(f"**{name} portfolio Performance:**")
    
    # Display performance metrics
    metrics_str = " | ".join([f"{index}: {100 * value:.2f}%" for index, value in perf.items()])
    st.write(metrics_str)
    
    st.write("**Weights:**")
    # Display weights
    weights_str = " | ".join([f"{x}: {100*y:.2f}%" for x, y in zip(assets, weights)])
    st.write(weights_str)



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
    