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

else:
    st.write("""
    ### User Manual: Technical Analysis Menu

    ---

    #### **Welcome to the Technical Analysis Menu**

    This section of the application allows you to perform in-depth technical analysis on stocks from various market indices. Before starting your analysis, please review the instructions below to ensure a smooth experience.

    ---

    ### **How to Use the Technical Analysis Menu**

    #### **Step 1: Configure Stock Parameters**
    1. **Select a Market Index**:
    - Use the dropdown menu in the sidebar to choose a market index (e.g., S&P500, CAC40, DAX, FTSE100, Nikkei225).  
    - The tickers for companies in the selected index will populate the next dropdown automatically.

    2. **Choose a Ticker**:
    - Pick a company ticker from the dropdown list. The company name will appear for easy identification.

    3. **Set the Date Range**:
    - Choose a **start date** and **end date** for the analysis period.
    - Ensure the start date is earlier than the end date. If not, an error message will prompt you to adjust the dates.

    ---

    #### **Step 2: Define Technical Analysis Parameters**
    1. **Volume (Optional)**:
    - Check the "Add volume" box to include trading volume in the analysis chart.

    2. **Simple Moving Average (SMA)**:
    - Expand the "SMA" section and:
        - Enable SMA by checking the box.
        - Adjust the period (default: 20).

    3. **Bollinger Bands (BB)**:
    - Expand the "Bollinger Bands" section and:
        - Enable BB by checking the box.
        - Adjust the periods and the number of standard deviations (default: 20 periods, 2 standard deviations).

    4. **Relative Strength Index (RSI)**:
    - Expand the "RSI" section and:
        - Enable RSI by checking the box.
        - Configure the RSI period, upper limit, and lower limit (default: 20 periods, 70 upper, 30 lower).

    ---

    #### **Step 3: Run Analysis**
    1. Click the **"Run Analysis"** button in the sidebar.
    2. The application will:
    - Retrieve stock data for the selected ticker and date range.
    - Display a preview of the data.
    - Generate an interactive technical analysis chart with the parameters you selected.

    ---

    ### **Note for First-Time Users**
    - The manual is visible until you click "Run Analysis."
    - Ensure all parameters are correctly configured before running the analysis to avoid errors.

    ---

    Enjoy exploring technical analysis with this tool! If you encounter any issues, double-check your inputs or reach out for assistance. 😊
    """)
