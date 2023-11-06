import streamlit as st

#my_var = "This is a variable from Home Page"
def main():
    # st.header("HOME PAGE")
    st.title("Financial Market Trading web app")
    st.write(" [Author: Josué AFOUDA](https://www.linkedin.com/in/josu%C3%A9-afouda/)")
    #st.write(my_var)

    choix = st.sidebar.radio("SubMenu", ["Description", "Documentation"])
    if choix == "Description":
        st.subheader("App description")
    if choix == "Documentation":
        st.subheader("Complete App documentation")
        st.write("""
            **I/ Technical Analysis Tool User Manual**

            Welcome to the Technical Analysis Tool. This user manual will guide you on how to effectively use this application for analyzing the performance of companies in the S&P500, CAC40, FTSE100, DAX and NIKKEI 225 index.

            **0. Select a Market index:**
            
            You can choose a stock market index from among the five most important in the world, namely: S&P500 (USA), CAC40 (Paris), FTSE100 (London), DAX (Frankfurt) and NIKKEI 225 (Tokyo).
                 
            **1. Select a Company:**

            You have the flexibility to choose any company that is a component of the the stock market index you previously chose. This company will be the focus of your stock data analysis.

            **2. Choose a Time Period:**

            Select a specific time period of your interest. This feature allows you to concentrate on the data that matters most to you.

            **3. Download Data:**

            You can download the selected stock data as a CSV file. This functionality is useful for further analysis or record-keeping.

            **4. Add Technical Indicators:**

            Enhance your analysis by incorporating technical indicators into the plot. The following indicators are available:

            - **Simple Moving Average (SMA):** This indicator helps you identify trends in the stock's price movements over time.

            - **Bollinger Bands:** Bollinger Bands provide insights into the stock's volatility and potential reversal points.

            - **Relative Strength Index (RSI):** RSI is a momentum indicator that can help you assess overbought or oversold conditions.

            **5. Customize Indicator Parameters:**

            Experiment with different parameters for the selected technical indicators. This feature enables you to fine-tune your analysis based on your preferences and trading strategy.

            By utilizing these features, you can perform a comprehensive technical analysis of your chosen company's stock data and make well-informed investment decisions.

            Feel free to explore the tool, and if you have any questions or require assistance, please do not hesitate to contact our support team (afouda.josue@gmail.com). Happy analyzing!
            """)


if __name__ == '__main__':
    main()