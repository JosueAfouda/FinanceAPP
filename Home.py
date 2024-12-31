import streamlit as st
import csv
import os

# Nom du fichier CSV
CSV_FILE = "submissions.csv"

# Vérifie si le fichier existe, sinon le crée avec un en-tête
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Email", "Project Type", "Timeline", "Budget", "Message"])

# Fonction pour ajouter une ligne dans le fichier CSV
def save_to_csv(data):
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(data)


#my_var = "This is a variable from Home Page"
def main():
    # st.header("HOME PAGE")

    # Interface utilisateur avec Streamlit
    st.set_page_config(page_title="Financial Market Trading web app", layout="wide")

    # Conteneur pour aligner les éléments horizontalement
    col1, col2, col3 = st.columns([1, 4, 1])

    # Colonne gauche : Image
    with col1:
        st.image(
            "linkedin_profil.png",  # Remplacez par le chemin de votre image
            width=80,     # Ajustez la taille si nécessaire
            use_column_width=False,
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

    #st.title("Financial Market Trading web app")
    #st.write(" [Author: Josué AFOUDA](https://www.linkedin.com/in/josu%C3%A9-afouda/)")
    #st.write(my_var)

    # Sidebar menu
    choix = st.sidebar.radio("SubMenu", ["Documentation", "Contact"])

    if choix == "Contact":
        st.subheader("Have a project in mind? Tell me about it below, and let's get started!")
        
        with st.form(key="contact_form"):
            name = st.text_input("Name", placeholder="Your Name")
            email = st.text_input("Email", placeholder="Your Email")
            project_type = st.selectbox(
                "Project Type",
                ["Select a project type", "Dashboard", "Workflow Automation", "Data Analytics", 
                "Predictive Modeling", "Data Cleaning", "Machine Learning Models", "Custom Data Applications"],
            )
            timeline = st.text_input("Timeline", placeholder="e.g., 2 weeks")
            budget = st.text_input("Budget", placeholder="Your Budget (optional)")
            message = st.text_area("Project Description", placeholder="Describe your project...")
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                if not name or not email or project_type == "Select a project type" or not timeline or not message:
                    st.error("Please fill in all the required fields!")
                else:
                    # Sauvegarder les données dans le fichier CSV
                    save_to_csv([name, email, project_type, timeline, budget, message])
                    st.success("Thank you for your submission! I'll get back to you shortly.")

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