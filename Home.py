import streamlit as st

st.set_page_config(page_title="Home | COVID Analytics", layout="wide",page_icon="image/covid_icon.png")

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
st.write("This system focuses not just on prediction, but on evaluating model reliability in uncertain time-series data.")
st.markdown("---")

st.markdown("## Problem Statement")

st.info("""
Time-series forecasting of COVID-19 cases to analyze short-term vs long-term prediction reliability.

This system compares multiple models to understand their behavior over time and evaluate how reliable they are under real-world conditions.
""")
st.link_button("View Source Code on Github", "https://github.com/sonudev-arch/covid19-forecast-dashboard")
st.markdown("## Project Overview")

st.write("""
This application analyzes COVID-19 trends and predicts future cases using multiple machine learning models.

The application is structured as a multi-page dashboard, where each page focuses on a specific analytical task.
""")
st.markdown("### Why This Project Matters")

st.write("""
Accurate forecasting during a pandemic is challenging due to uncertainty, noise, and changing patterns.

This project explores how different models behave under such conditions, highlighting their strengths and limitations in real-world scenarios.
""")

st.markdown("---")

st.markdown("## Objective")

st.info("""
- Analyze pandemic trends over time  
- Detect and compare multiple waves  
- Understand growth patterns and volatility  
- Evaluate vaccination impact  
- Compare forecasting models for short-term vs long-term accuracy  
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
        "Comprehensive exploration of multiple COVID-19 datasets including structure, data quality, statistical summaries, and preprocessing insights for time-series analysis.",
        "2_Datasets.py"
    )

    page_card(
        "Data Analysis", 
        "Exploratory analysis of time-series trends, moving averages, growth    patterns, and inter-state comparisons.",
        "3_Data_Analysis.py"
    )

    page_card(
        "Time Decomposition",
        "Breakdown of time-series into trend, seasonality, and residuals with stationarity testing for model readiness.",
        "4_Time_Decomposition.py"
    )


with col2:
 

    page_card(
        "Models",
        "Understanding assumptions, strengths, and limitations of ARIMA, Holt-Winters, SVR, and Fourier.",
        "5_Models.py"
    )

    page_card(
        "Forecasts",
        "Model predictions, performance comparison, and evaluation of short-term vs long-term forecast accuracy.",
        "6_Forecasts.py"
    )
    page_card(
        "About",
        "Overview of project motivation, methodology, and insights from time-series forecasting and model evaluation.",
        "About.py"
    )
st.markdown("---")

st.markdown("## Key Insight")

st.success("""
Different models behave differently across time horizons. 
A model that performs well in short-term forecasting may fail in long-term predictions due to trend shifts and noise.
""")
st.markdown("## ML Pipeline and Validation Strategy")

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