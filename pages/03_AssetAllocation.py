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

import plotly.io as pio

# Définir une configuration globale pour les graphiques
pio.templates["custom_template"] = pio.templates["plotly_dark"]  # Ou "plotly" si vous n'utilisez pas de thème sombre
pio.templates["custom_template"].layout.legend = dict(
    orientation="h",  # Orientation horizontale
    yanchor="bottom",  # Ancrer la légende en bas
    y=1.02,  # Position verticale légèrement au-dessus du graphique
    xanchor="center",  # Centrer horizontalement
    x=0.5  # Position horizontale au milieu
)

# Appliquer le template globalement
pio.templates.default = "custom_template"

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
st.sidebar.header("Asset Allocation Parameters")

rf_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0, step=0.1) / 100

allocation_method = st.sidebar.radio("Select Asset Allocation Technique", 
                                     ["CVXPY optimization", "Scipy optimization", "Monte Carlo simulations", "Hierarchical Risk Parity"],
                                     index=0)

st.sidebar.markdown("---")
st.sidebar.info("**Disclaimer:** Past performance is not indicative of future results. Portfolio optimization is based on historical data and does not guarantee future returns.")

#if len(assets) > 0:
    #assets_data = load_data(assets, start_date, end_date)['Adj Close']
#else:
    #st.sidebar.write("Choose some assets to build your Portfolio")

run_allocation_button = st.sidebar.button("Run Asset Allocation")

if run_allocation_button:
    if len(assets) > 0:
        assets_data = load_data(assets, start_date, end_date)['Adj Close']
        
        # Warning for data loss due to differing asset histories
        initial_len = len(assets_data)
        returns_df = assets_data.pct_change().dropna()
        final_len = len(returns_df)
        if final_len < initial_len * 0.5:
            st.warning(f"Significant data loss detected: {initial_len - final_len} days removed due to differing asset histories. The analysis is based on {final_len} common trading days.")

        # Créer les colonnes pour afficher les graphiques côte à côte
        col1, col2 = st.columns(2)

        # Graphique des prix ajustés
        fig = go.Figure()
        for asset in assets:
            fig.add_trace(go.Scatter(x=assets_data.index, y=assets_data[asset], mode='lines', name=tickers_companies_dict[asset]))

        fig.update_layout(
            title="Adjusted Close Prices of Selected Stocks",
            xaxis_title="Date",
            yaxis_title="Adjusted Close Price",
            legend=dict(title="Stocks")
        )

        # Afficher le graphique des prix ajustés dans la première colonne
        with col1:
            st.plotly_chart(fig)

        # Calculer et afficher les rendements quotidiens
        avg_returns, cov_mat = calculate_statistics(assets_data)

        fig_returns = go.Figure()
        for asset in assets:
            fig_returns.add_trace(go.Scatter(x=assets_data.index, y=returns_df[asset], mode='lines', name=tickers_companies_dict[asset]))

        fig_returns.update_layout(
            title="Daily Returns of Selected Stocks",
            xaxis_title="Date",
            yaxis_title="Daily Returns",
            legend=dict(title="Company Names")
        )

        # Afficher le graphique des rendements quotidiens dans la deuxième colonne
        with col2:
            st.plotly_chart(fig_returns)

        # Simulate random portfolio weights
        np.random.seed(42)
        N_PORTFOLIOS = 20000
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
        portf_sharpe_ratio = (portf_rtns - rf_rate) / portf_vol

        # Create a DataFrame containing all the data
        portf_results_df = pd.DataFrame(
            {"returns": portf_rtns,
            "volatility": portf_vol,
            "sharpe_ratio": portf_sharpe_ratio}
        )

        display_data_preview("Preview data", portf_results_df, key=6)
            

        if allocation_method == "Monte Carlo simulations":
            
            # Locate the points creating the Simulated Boundary
            N_POINTS = 100
            ef_rtn_list = []
            ef_vol_list = []

            possible_ef_rtns = np.linspace(
                portf_results_df["returns"].min(), 
                portf_results_df["returns"].max(), 
                N_POINTS
            )
            possible_ef_rtns = np.round(possible_ef_rtns, 2)    
            portf_rtns_rounded = np.round(portf_rtns, 2)

            for rtn in possible_ef_rtns:
                if rtn in portf_rtns_rounded:
                    ef_rtn_list.append(rtn)
                    matched_ind = np.where(portf_rtns_rounded == rtn)
                    ef_vol_list.append(np.min(portf_vol[matched_ind]))

            # Create the Simulated Boundary plot using Matplotlib
            st.subheader("Simulated Boundary (Monte Carlo)")
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
            ax_ef.set(xlabel="Volatility", ylabel="Expected Returns", title="Simulated Boundary - Monte Carlo")

            # Line plot for Simulated Boundary
            ax_ef.plot(ef_vol_list, ef_rtn_list, "b--", label="Simulated Frontier")

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
            cbar.set_label(f"Sharpe Ratio (RF={rf_rate*100:.1f}%)")

            # Add legend
            ax_ef.legend()

            # Remove spines and tighten layout
            sns.despine()
            plt.tight_layout()

            # Display the Simulated Boundary chart
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
            weight_chart = px.bar(
                weight_chart_data,
                x="Assets",
                y="Weights",
                labels={"Weights": "Weight"},
                title="Asset Weights in Maximum Sharpe Ratio Portfolio",
                color="Assets"
            )

            # Créer les colonnes pour afficher le graphique et le résumé côte à côte
            col1, col2 = st.columns(2)

            # Afficher le graphique des poids des actifs dans la première colonne
            with col1:
                st.plotly_chart(weight_chart)

            # Afficher le résumé du portefeuille dans la deuxième colonne
            with col2:
                print_portfolio_summary(perf=max_sharpe_portf, weights=weights[max_sharpe_ind], assets=assets, name="Maximum Sharpe Ratio")


            # Minimum Volatility Portfolio
            weight_chart_data2 = pd.DataFrame({"Assets": assets, "Weights": weights[min_vol_ind]}).sort_values(by="Weights", ascending=False)
            weight_chart2 = px.bar(
                weight_chart_data2,
                x="Assets",
                y="Weights",
                labels={"Weights": "Weight"},
                title="Asset Weights in Minimum Volatility Portfolio",
                color="Assets"
            )

            # Créer deux colonnes pour afficher le graphique et le résumé côte à côte
            col1, col2 = st.columns(2)

            # Afficher le graphique dans la première colonne
            with col1:
                st.plotly_chart(weight_chart2)

            # Afficher le résumé du portefeuille dans la deuxième colonne
            with col2:
                print_portfolio_summary(min_vol_portf, weights[min_vol_ind], assets, name="Minimum Volatility")


            # Maximum Return Portfolio
            weight_chart_data3 = pd.DataFrame({"Assets": assets, "Weights": weights[max_return_ind]}).sort_values(by="Weights", ascending=False)
            weight_chart3 = px.bar(
                weight_chart_data3,
                x="Assets",
                y="Weights",
                labels={"Weights": "Weight"},
                title="Asset Weights in Maximum Return Portfolio",
                color="Assets"
            )

            # Créer deux colonnes pour afficher le graphique et le résumé côte à côte
            col1, col2 = st.columns(2)

            # Afficher le graphique dans la première colonne
            with col1:
                st.plotly_chart(weight_chart3)

            # Afficher le résumé du portefeuille dans la deuxième colonne
            with col2:
                print_portfolio_summary(perf=max_return_portf, weights=weights[max_return_ind], assets=assets, name="Maximum Return")


        elif allocation_method == "Scipy optimization":

            # Dynamic return range based on asset performance
            rtns_range = np.linspace(avg_returns.min(), avg_returns.max(), 100)
            
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
                            title=f"Efficient Frontier - SciPy Optimization (RF={rf_rate*100:.1f}%)")

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
                "Sharpe Ratio": ((min_vol_portf_rtn_scipy - rf_rate) / min_vol_portf_vol_scipy)
            }
            #st.write(min_vol_portf_scipy)
            weight_chart_data4 = pd.DataFrame({"Assets": assets, "Weights": efficient_portfolios_scipy[min_vol_ind_scipy]["x"]}).sort_values(by="Weights", ascending=False)
            weight_chart4 = px.bar(weight_chart_data4, x="Assets", y="Weights", labels={"Weights": "Weight"}, 
                                title="Asset Weights in Minimum Volatility Portfolio", color="Assets")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(weight_chart4)
            with col2:
                print_portfolio_summary(min_vol_portf_scipy, 
                            efficient_portfolios_scipy[min_vol_ind_scipy]["x"], 
                            assets, 
                            name="Minimum Volatility")


            # Maximum Sharpe Ratio Portfolio
            n_assets = len(avg_returns)

            args = (avg_returns, cov_mat, rf_rate)
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
            with col1:
                st.plotly_chart(weight_chart5)
            with col2:
                print_portfolio_summary(max_sharpe_portf_scipy, max_sharpe_portf_w_scipy, assets, name="Maximum Sharpe Ratio")

    
        elif allocation_method == "CVXPY optimization":
            avg_returns_val = avg_returns.values
            cov_mat_val = cov_mat.values

            # Set up the optimization problem
            weights_var = cp.Variable(n_assets)
            gamma_par = cp.Parameter(nonneg=True)
            portf_rtn_cvx = avg_returns_val @ weights_var 
            portf_var_cvx = cp.quad_form(weights_var, cov_mat_val) # This is Variance
            objective_function = cp.Maximize(
                portf_rtn_cvx - gamma_par * portf_var_cvx
            )
            problem = cp.Problem(
                objective_function, 
                [cp.sum(weights_var) == 1, weights_var >= 0]
            )

            # Calculate the Efficient Frontier
            N_POINTS_CVX = 25
            portf_rtn_cvx_ef = []
            portf_vol_cvx_ef = []
            weights_ef = []
            gamma_range = np.logspace(-3, 3, num=N_POINTS_CVX)

            for gamma in gamma_range:
                gamma_par.value = gamma
                problem.solve()
                portf_vol_cvx_ef.append(cp.sqrt(portf_var_cvx).value)
                portf_rtn_cvx_ef.append(portf_rtn_cvx.value)
                weights_ef.append(weights_var.value)

            # Plot the Efficient Frontier, together with the individual assets
            fig_cvx, ax_cvx = plt.subplots()
            MARKERS = generate_markers(n_assets)
            ax_cvx.plot(portf_vol_cvx_ef, portf_rtn_cvx_ef, "g-", label="Efficient Frontier (CVXPY)")
            for asset_index in range(n_assets):
                plt.scatter(x=np.sqrt(cov_mat_val[asset_index, asset_index]), 
                            y=avg_returns_val[asset_index], 
                            marker=MARKERS[asset_index], 
                            label=assets[asset_index],
                            s=150)
            ax_cvx.set(title="Efficient Frontier - CVXPY Optimization",
                xlabel="Volatility", 
                ylabel="Expected Returns")
            ax_cvx.legend()

            sns.despine()
            plt.tight_layout()
            st.pyplot(fig_cvx)


        elif allocation_method == "Hierarchical Risk Parity":
            # Calculate HRP weights
            hrp_weights = get_hrp_weights(assets_data)
            
            # Organize data for display (ensure alignment with asset list if needed, but HRP returns sorted Series)
            # Reindex to match the user's selected 'assets' list order for consistency in visualization
            hrp_weights = hrp_weights.reindex(assets) 
            
            # Calculate metrics
            hrp_ret = get_portf_rtn(hrp_weights.values, avg_returns) # Ensure .values for numpy operations if avg_returns is Series
            hrp_vol = get_portf_vol(hrp_weights.values, avg_returns, cov_mat)
            hrp_sharpe = (hrp_ret - rf_rate) / hrp_vol
            
            hrp_perf = pd.Series({
                "Return": hrp_ret,
                "Volatility": hrp_vol,
                "Sharpe Ratio": hrp_sharpe
            })

            # Create columns
            col1, col2 = st.columns(2)
            
            # Chart
            weight_chart_data_hrp = pd.DataFrame({"Assets": hrp_weights.index, "Weights": hrp_weights.values}).sort_values(by="Weights", ascending=False)
            weight_chart_hrp = px.bar(
                weight_chart_data_hrp,
                x="Assets",
                y="Weights",
                labels={"Weights": "Weight"},
                title="Asset Weights (HRP)",
                color="Assets"
            )
            
            with col1:
                st.plotly_chart(weight_chart_hrp)
                
            with col2:
                print_portfolio_summary(hrp_perf, hrp_weights.values, hrp_weights.index, name="HRP")

            st.info("Hierarchical Risk Parity (HRP) builds a diversified portfolio by clustering assets based on correlation and allocating risk recursively. It does not require inverting the covariance matrix, making it more robust to noise and outliers than traditional Mean-Variance Optimization.")
    else:
        st.sidebar.write("Choose some assets to build your Portfolio")

else:
    st.write("""
    ## Asset Allocation Application - User Manual

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
