import streamlit as st

st.set_page_config(page_title="Home | COVID Analytics", layout="wide")

st.markdown("""
<style>
.card {
    background-color: #111;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #222;
    margin-bottom: 15px;
}

.card-title {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 8px;
}

.card-text {
    font-size: 14px;
    color: #bbb;
}
</style>
""", unsafe_allow_html=True)

st.title("COVID-19 Analytics & Forecasting System")
st.caption("End-to-end time-series analysis, wave detection, and forecasting of COVID-19 in India")

st.markdown("---")

st.markdown("## Project Overview")

st.write("""
This project is a complete data science system designed to analyze the spread and impact of COVID-19.
It combines time-series analysis, statistical modeling, and machine learning techniques to extract insights,
identify patterns, and generate future forecasts.

The application is structured as a multi-page dashboard, where each page focuses on a specific analytical task.
""")

st.markdown("---")

st.markdown("## Objective")

st.info("""
- Analyze pandemic trends over time  
- Detect and compare multiple waves  
- Understand growth patterns and volatility  
- Evaluate vaccination impact  
- Forecast future scenarios using multiple models  
""")

st.markdown("---")

st.markdown("## Explore Dashboard Sections")

def page_card(title, desc, page_file):
    st.markdown(f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-text">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(f"Open {title}", key=title):
        st.switch_page(f"pages/{page_file}")

col1, col2 = st.columns(2)

with col1:
    page_card(
        "Datasets",
        "Datasets, Data Info, Data Quality and Data Preview",
        "3_Data_Analysis.py"
    )

    page_card(
        "Trend Analysis",
        "Time-series trends, moving averages, and state comparisons.",
        "3_Data_Analysis.py"
    )

    page_card(
        "Time Decomposition",
        "Trend, seasonality, residual analysis + stationarity testing.",
        "4_Time_Decomposition.py"
    )


with col2:
 

    page_card(
        "Models",
        "Overview of ARIMA, Holt-Winters, SVR, and Fourier models.",
        "5_Models.py"
    )

    page_card(
        "Predictions",
        "Forecasting, model comparison, and future wave prediction.",
        "6_Forecasts.py"
    )
    page_card(
        "About",
        "About",
        "About.py"
    )
st.markdown("---")

st.markdown("## Analytical Pipeline")

st.markdown("""
1. Data Collection & Cleaning  
2. Exploratory Data Analysis  
3. Feature Engineering (Active, Growth)  
4. Time-Series Decomposition  
5. Stationarity Testing (ADF)  
6. Model Training & Walk-forward Validation  
7. Forecast Generation & Evaluation  
""")

st.markdown("---")


st.markdown("## Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Languages & Tools**")
    st.write("- Python")
    st.write("- Streamlit")

with col2:
    st.write("**Libraries**")
    st.write("- Pandas, NumPy")
    st.write("- Plotly")
    st.write("- Scikit-learn")
    st.write("- Statsmodels")

with col3:
    st.write("**Techniques**")
    st.write("- Time-Series Analysis")
    st.write("- Peak Detection")
    st.write("- Statistical Modeling")
    st.write("- Machine Learning")

st.markdown("---")


st.warning("""
This project uses historical COVID-19 data.
All insights and forecasts are based on past trends and assumptions and are not real-time predictions.
""")

st.success("""
This system demonstrates how data science techniques can transform raw data into meaningful insights,
helping understand past patterns and anticipate future trends.
""")