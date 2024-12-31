# imports
import streamlit as st
import random
import numpy as np
import scipy.optimize as sco
import cvxpy as cp
import yfinance as yf
import datetime 
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from utils import *

st.title("Asset Allocation")

#st.subheader('Technical Analysis Page')
# df = "ma dataframe"
#from Home import my_var

#st.write(my_var)

## set offline mode for cufflinks
cf.go_offline()


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

start_date = st.sidebar.date_input(
    "Start date", 
    datetime.date(2019, 1, 1)
)
end_date = st.sidebar.date_input(
    "End date", 
    datetime.date.today()
)

assets = st.sidebar.multiselect(
    'Select the assets for the portfolio:', 
    available_tickers, 
    #default=random.sample(available_tickers, 3),
    format_func=tickers_companies_dict.get
)

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

## inputs for Asset Allocation analysis
st.sidebar.header("Asset Allocation Method")

allocation_method = st.sidebar.radio("Select Asset Allocation Technique", 
                                     ["Monte Carlo simulations", "Scipy optimization", "CVXPY optimization", "Hierarchical Risk Parity"])

#if len(assets) > 0:
    #assets_data = load_data(assets, start_date, end_date)['Adj Close']
#else:
    #st.sidebar.write("Choose some assets to build your Portfolio")

run_allocation_button = st.sidebar.button("Run Asset Allocation")

if run_allocation_button:
    if len(assets) > 0:
        assets_data = load_data(assets, start_date, end_date)['Adj Close']
        # Plot stock prices
        fig = go.Figure()

        for asset in assets:
            fig.add_trace(go.Scatter(x=assets_data.index, y=assets_data[asset], mode='lines', name=tickers_companies_dict[asset]))

        fig.update_layout(title="Adjusted Close Prices of Selected Stocks",
                        xaxis_title="Date",
                        yaxis_title="Adjusted Close Price",
                        legend=dict(title="Stocks"))

        # Display the plot
        st.plotly_chart(fig)

        # Calculate and display daily returns
        avg_returns, cov_mat = calculate_statistics(assets_data)

        fig_returns = go.Figure()

        for asset in assets:
            fig_returns.add_trace(go.Scatter(x=assets_data.index, y=assets_data[asset].pct_change(), mode='lines', name=tickers_companies_dict[asset]))

        fig_returns.update_layout(title="Daily Returns of Selected Stocks",
                                xaxis_title="Date",
                                yaxis_title="Daily Returns",
                                legend=dict(title="Company Names"))

        # Display the plot
        st.plotly_chart(fig_returns)

        # Simulate random portfolio weights
        np.random.seed(42)
        N_PORTFOLIOS = 10 ** 5
        n_assets = len(assets)
        weights = np.random.random(size=(N_PORTFOLIOS, n_assets))
        weights /= np.sum(weights, axis=1)[:, np.newaxis]

        # Calculate the portfolio metrics
        portf_rtns = np.dot(weights, avg_returns)
        portf_vol = []
        for i in range(0, len(weights)):
            vol = np.sqrt(np.dot(weights[i].T, np.dot(cov_mat, weights[i])))
            portf_vol.append(vol)
        portf_vol = np.array(portf_vol)
        portf_sharpe_ratio = portf_rtns / portf_vol

        # Create a DataFrame containing all the data
        portf_results_df = pd.DataFrame(
            {"returns": portf_rtns,
            "volatility": portf_vol,
            "sharpe_ratio": portf_sharpe_ratio}
        )

        display_data_preview("Preview data", portf_results_df, key=6)
            

        if allocation_method == "Monte Carlo simulations":
            
            # Locate the points creating the Efficient Frontier
            N_POINTS = 100
            ef_rtn_list = []
            ef_vol_list = []

            possible_ef_rtns = np.linspace(
                portf_results_df["returns"].min(), 
                portf_results_df["returns"].max(), 
                N_POINTS
            )
            possible_ef_rtns = np.round(possible_ef_rtns, 2)    
            portf_rtns = np.round(portf_rtns, 2)

            for rtn in possible_ef_rtns:
                if rtn in portf_rtns:
                    ef_rtn_list.append(rtn)
                    matched_ind = np.where(portf_rtns == rtn)
                    ef_vol_list.append(np.min(portf_vol[matched_ind]))

            # Create the Efficient Frontier plot using Matplotlib
            st.subheader("Efficient Frontier")
            fig_ef, ax_ef = plt.subplots(figsize=(10, 6))

            # Scatter plot for individual portfolios
            scatter = ax_ef.scatter(
                x=portf_results_df["volatility"],
                y=portf_results_df["returns"],
                c=portf_results_df["sharpe_ratio"],
                cmap="RdYlGn",
                edgecolors="black",
                marker="o",
                alpha=0.8,
            )
            ax_ef.set(xlabel="Volatility", ylabel="Expected Returns", title="Efficient Frontier")

            # Line plot for Efficient Frontier
            ax_ef.plot(ef_vol_list, ef_rtn_list, "b--")

            # Markers for individual assets
            MARKERS = generate_markers(n_assets)
            for asset_index in range(n_assets):
                ax_ef.scatter(
                    x=np.sqrt(cov_mat.iloc[asset_index, asset_index]),
                    y=avg_returns[asset_index],
                    marker=MARKERS[asset_index],
                    s=150,
                    color="black",
                    label=tickers_companies_dict[assets[asset_index]],
                )

            # Add colorbar
            cbar = fig_ef.colorbar(scatter)
            cbar.set_label("Sharpe Ratio")

            # Add legend
            ax_ef.legend()

            # Remove spines and tighten layout
            sns.despine()
            plt.tight_layout()

            # Display the Efficient Frontier chart
            st.pyplot(fig_ef)

            # Display the portfolio performance summary
            st.subheader("Portfolio Performance Summary")
            max_sharpe_ind = np.argmax(portf_results_df["sharpe_ratio"])
            max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]

            min_vol_ind = np.argmin(portf_results_df["volatility"])
            min_vol_portf = portf_results_df.loc[min_vol_ind]

            max_return_ind = np.argmax(portf_results_df["returns"])
            max_return_portf = portf_results_df.loc[max_return_ind]

            # Bar chart showing the calculated weight of each asset in the Maximum Sharpe Ratio portfolio
            # Maximum Sharpe Ratio Portfolio
            weight_chart_data = pd.DataFrame({"Assets": assets, "Weights": weights[max_sharpe_ind]}).sort_values(by="Weights", ascending=False)
            weight_chart = px.bar(weight_chart_data, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                                title="Asset Weights in Maximum Sharpe Ratio Portfolio", color="Assets")
            st.plotly_chart(weight_chart)
            print_portfolio_summary(perf=max_sharpe_portf, weights=weights[max_sharpe_ind], assets=assets, name="Maximum Sharpe Ratio")

            # Minimum Volatility Portfolio
            weight_chart_data2 = pd.DataFrame({"Assets": assets, "Weights": weights[min_vol_ind]}).sort_values(by="Weights", ascending=False)
            weight_chart2 = px.bar(weight_chart_data2, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                                title="Asset Weights in Minimum Volatility Portfolio", color="Assets")
            st.plotly_chart(weight_chart2)
            print_portfolio_summary(min_vol_portf, weights[min_vol_ind], assets, name="Minimum Volatility")

            # Maximum Return Portfolio
            weight_chart_data3 = pd.DataFrame({"Assets": assets, "Weights": weights[max_return_ind]}).sort_values(by="Weights", ascending=False)
            weight_chart3 = px.bar(weight_chart_data3, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                                title="Asset Weights in Maximum Return Portfolio", color="Assets")
            st.plotly_chart(weight_chart3)
            print_portfolio_summary(perf=max_return_portf, weights=weights[max_return_ind], assets=assets, name="Maximum Return")


        elif allocation_method == "Scipy optimization":

            rtns_range = np.linspace(-0.1, 0.55, 200)
            
            # Calculate the Efficient Frontier using SciPy optimization
            efficient_portfolios_scipy = get_efficient_frontier_scipy(avg_returns, cov_mat, rtns_range)

            # Extract the volatilities of the efficient portfolios
            vols_range_scipy = [x["fun"] for x in efficient_portfolios_scipy]

            # Plot the Efficient Frontier using SciPy optimization
            with sns.plotting_context("paper"):
                fig_scipy, ax_scipy = plt.subplots()
                portf_results_df.plot(kind="scatter", x="volatility",
                                    y="returns", c="sharpe_ratio",
                                    cmap="RdYlGn", edgecolors="black",
                                    ax=ax_scipy)
                ax_scipy.plot(vols_range_scipy, rtns_range, "b--", linewidth=3)
                ax_scipy.set(xlabel="Volatility",
                            ylabel="Expected Returns",
                            title="Efficient Frontier - SciPy Optimization")

                sns.despine()
                plt.tight_layout()

            # Display the Efficient Frontier chart using SciPy optimization
            st.pyplot(fig_scipy)

            # Minimum Volatility Portfolio
            min_vol_ind_scipy = np.argmin(vols_range_scipy)
            min_vol_portf_rtn_scipy = rtns_range[min_vol_ind_scipy]
            min_vol_portf_vol_scipy = efficient_portfolios_scipy[min_vol_ind_scipy]["fun"]

            min_vol_portf_scipy = {
                "Return": min_vol_portf_rtn_scipy,
                "Volatility": min_vol_portf_vol_scipy,
                "Sharpe Ratio": (min_vol_portf_rtn_scipy / min_vol_portf_vol_scipy)
            }
            #st.write(min_vol_portf_scipy)
            weight_chart_data4 = pd.DataFrame({"Assets": assets, "Weights": efficient_portfolios_scipy[min_vol_ind_scipy]["x"]}).sort_values(by="Weights", ascending=False)
            weight_chart4 = px.bar(weight_chart_data4, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                                title="Asset Weights in Minimum Volatility Portfolio", color="Assets")
            st.plotly_chart(weight_chart4)
            print_portfolio_summary(min_vol_portf_scipy, 
                            efficient_portfolios_scipy[min_vol_ind_scipy]["x"], 
                            assets, 
                            name="Minimum Volatility")


            # Maximum Sharpe Ratio Portfolio
            n_assets = len(avg_returns)
            RF_RATE = 0

            args = (avg_returns, cov_mat, RF_RATE)
            constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
            bounds = tuple((0,1) for asset in range(n_assets))
            initial_guess = n_assets * [1. / n_assets]

            max_sharpe_portf_scipy = sco.minimize(neg_sharpe_ratio, 
                                            x0=initial_guess, 
                                            args=args,
                                            method="SLSQP", 
                                            bounds=bounds, 
                                            constraints=constraints)
            
            max_sharpe_portf_w_scipy = max_sharpe_portf_scipy["x"]
            max_sharpe_portf_scipy = {
                "Return": get_portf_rtn(max_sharpe_portf_w_scipy, avg_returns),
                "Volatility": get_portf_vol(max_sharpe_portf_w_scipy, 
                                            avg_returns, 
                                            cov_mat),
                "Sharpe Ratio": -max_sharpe_portf_scipy["fun"]
            }
            #st.write(max_sharpe_portf)
            weight_chart_data5 = pd.DataFrame({"Assets": assets, "Weights": max_sharpe_portf_w_scipy}).sort_values(by="Weights", ascending=False)
            weight_chart5 = px.bar(weight_chart_data5, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                                title="Asset Weights in Maximum Sharpe Ratio Portfolio", color="Assets")
            st.plotly_chart(weight_chart5)
            print_portfolio_summary(max_sharpe_portf_scipy, max_sharpe_portf_w_scipy, assets, name="Maximum Sharpe Ratio")

    

        elif allocation_method == "CVXPY optimization":
            avg_returns = avg_returns.values
            cov_mat = cov_mat.values

            # Set up the optimization problem
            weights = cp.Variable(n_assets)
            gamma_par = cp.Parameter(nonneg=True)
            portf_rtn_cvx = avg_returns @ weights 
            portf_vol_cvx = cp.quad_form(weights, cov_mat)
            objective_function = cp.Maximize(
                portf_rtn_cvx - gamma_par * portf_vol_cvx
            )
            problem = cp.Problem(
                objective_function, 
                [cp.sum(weights) == 1, weights >= 0]
            )

            # Calculate the Efficient Frontier
            N_POINTS = 25
            portf_rtn_cvx_ef = []
            portf_vol_cvx_ef = []
            weights_ef = []
            gamma_range = np.logspace(-3, 3, num=N_POINTS)

            for gamma in gamma_range:
                gamma_par.value = gamma
                problem.solve()
                portf_vol_cvx_ef.append(cp.sqrt(portf_vol_cvx).value)
                portf_rtn_cvx_ef.append(portf_rtn_cvx.value)
                weights_ef.append(weights.value)

            # Plot the Efficient Frontier, together with the individual assets
            fig_cvx, ax_cvx = plt.subplots()
            MARKERS = generate_markers(n_assets)
            ax_cvx.plot(portf_vol_cvx_ef, portf_rtn_cvx_ef, "g-")
            for asset_index in range(n_assets):
                plt.scatter(x=np.sqrt(cov_mat[asset_index, asset_index]), 
                            y=avg_returns[asset_index], 
                            marker=MARKERS[asset_index], 
                            label=assets[asset_index],
                            s=150)
            ax_cvx.set(title="Efficient Frontier",
                xlabel="Volatility", 
                ylabel="Expected Returns")
            ax_cvx.legend()

            sns.despine()
            plt.tight_layout()
            st.pyplot(fig_cvx)


        elif allocation_method == "Hierarchical Risk Parity":
            st.write("----------------------------------------will be available soon-------------------------------------------")
    else:
        st.sidebar.write("Choose some assets to build your Portfolio")

else:
    st.write("""
    # Asset Allocation Application - User Manual

    Welcome to the Asset Allocation Application! This app allows you to analyze and optimize your investment portfolio using various techniques. Please follow the steps below to make the most of this tool.

    ## Instructions:

    1. **Select a Market Index**:
    - Use the sidebar to choose a market index (e.g., S&P500, CAC40, DAX, etc.). This will determine the available assets for your portfolio.

    2. **Choose a Date Range**:
    - Specify the start and end dates for historical data analysis. Ensure the start date is earlier than the end date.

    3. **Select Assets**:
    - Pick the assets you wish to include in your portfolio from the available list. You can select multiple assets.

    4. **Pick an Asset Allocation Method**:
    - Choose one of the following techniques for portfolio optimization:
        - **Monte Carlo Simulations**
        - **Scipy Optimization**
        - **CVXPY Optimization**
        - **Hierarchical Risk Parity** (Coming soon!)

    5. **Run the Analysis**:
    - Click on the "Run Asset Allocation" button in the sidebar to execute the analysis based on your selected parameters.

    ## Notes:
    - If no assets are selected, the app cannot perform the allocation.
    - For Monte Carlo and other optimization techniques, the app will generate and display:
    - Adjusted close prices of selected stocks.
    - Daily returns.
    - Portfolio performance metrics.
    - Efficient Frontier and portfolio allocation charts.

    ## Tips:
    - Experiment with different asset combinations and allocation methods to find the best fit for your investment goals.
    - Review the generated charts and tables for insights into portfolio performance and risk-return trade-offs.

    Enjoy exploring the world of asset allocation!

    """)
