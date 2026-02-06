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
    - **Download Model**: Check this box to download the trained Prophet model for further use.

    2. **Modeling Process**:
    - Select "Prophet" from the radio buttons to enable Prophet-based modeling.

    ---

    #### **Step 3: Run Forecasting**
    1. Click the **"Run Forecasting"** button in the sidebar.
    2. The application will:
    - Retrieve the stock data for the selected ticker and date range.
    - Display a preview of the data.
    - Plot the historical closing prices.
    - Generate forecasts using the configured Prophet parameters.

    ---

    ### **Output Visualizations and Data**
    1. **Close Price Chart**:
    - View an interactive chart of the historical closing prices.

    2. **Forecasts and Actual Prices**:
    - Examine side-by-side comparisons of actual vs. predicted prices.
    - Review uncertainty intervals for the predictions.

    3. **Changepoints Analysis**:
    - Visualize changepoints identified during modeling to understand shifts in stock behavior.

    4. **Future Forecasts**:
    - Explore forecasted prices for the specified horizon beyond the historical data range.

    5. **Download Model**:
    - If selected, download the trained Prophet model as a JSON file for future use.

    ---

    ### **Note for First-Time Users**
    - This manual will remain visible until you click "Run Forecasting."
    - Carefully configure all parameters to ensure accurate forecasting results.

    ---

    Enjoy forecasting with this tool! If you have any questions or encounter issues, double-check your inputs or seek assistance. 🚀
    """)
