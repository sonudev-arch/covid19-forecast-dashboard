import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.title("About")
st.caption("Understanding the motivation, methodology, and analytical depth of the COVID-19 Analytics System")

st.markdown("---")

st.markdown("## Purpose")



st.markdown("---")

st.markdown("## System Capabilities")

st.info("""
This system provides a complete analytical pipeline:

- Tracks and visualizes pandemic progression over time  
- Detects multiple waves using signal processing techniques  
- Analyzes growth dynamics and volatility patterns  
- Evaluates vaccination impact on case trends  
- Performs advanced exploratory data analysis (EDA)  
- Applies multiple forecasting models  
- Compares model performance using RMSE and MAE  
- Predicts future waves and peak intensities  
""")

st.markdown("---")

st.markdown("## Methodology and Analytical Pipeline")

st.write("""
The web app follows a structured and modular workflow, ensuring both interpretability and robustness:

1. Data Processing - cleaning and structuring  
2. EDA - trends, peaks, comparisons  
3. Feature Engineering - active cases, growth rate  
4. Decomposition - trend, seasonality, residual  
5. Testing - ADF and differencing  
6. Models - ARIMA, Holt-Winters, SVR, Fourier  
7. Evaluation - RMSE, MAE, walk-forward validation  
8. Forecasting - future trends and wave prediction  
""")

st.markdown("---")

st.markdown("## Key Highlights")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    Advanced Time-Series Analysis
    - Trend, seasonality, and residual decomposition  
    - Moving averages and smoothing techniques  
    - Stationarity testing using ADF  

    Wave Detection System
    - Automated peak detection using SciPy  
    - Identification of multiple pandemic waves  
    - Severity and duration comparison  
    """)

with col2:
    st.success("""
    Multi-Model Forecasting
    - ARIMA, Holt-Winters, SVR, Fourier  
    - Walk-forward validation  
    - Model comparison using RMSE and MAE  

    Real-World Integration
    - Vaccination-adjusted forecasting  
    - Impact-based suppression modeling  
    - Improved interpretability of predictions  
    """)

st.markdown("---")

st.markdown("## Analytical Strength of the System")

st.write("""
This system extends beyond basic dashboards by integrating:

- Signal processing for wave detection  
- Statistical modeling for trend analysis  
- Machine learning for predictive modeling  
- Hybrid approaches combining Fourier analysis with real-world vaccination effects  

This combination enables the system to capture long-term trends, cyclical patterns, and sudden changes,
resulting in a more realistic and interpretable forecasting framework.
""")

st.markdown("---")

st.markdown("## Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Languages and Tools")
    st.write("- Python")
    st.write("- Streamlit")

with col2:
    st.write("Libraries Used")
    st.write("- Pandas, NumPy")
    st.write("- Plotly")
    st.write("- Scikit-learn")
    st.write("- Statsmodels")
    st.write("- SciPy")

with col3:
    st.write("Techniques Applied")
    st.write("- Time-Series Analysis")
    st.write("- Machine Learning")
    st.write("- Signal Processing")
    st.write("- Statistical Modeling")

st.markdown("---")

st.markdown("## About the Developer")

st.write("""
This project was developed to apply data science concepts to a real-world problem.

It demonstrates practical skills in:
- Time-series analysis and forecasting  
- Data-driven reasoning and model evaluation  
- Building interactive analytical systems using Streamlit  

The focus throughout the project is on understanding model behavior,
not just generating predictions.
""")

st.markdown("---")

st.warning("""
This uses historical COVID-19 data.
All insights and forecasts are based on past trends and assumptions and should not be interpreted as real-time predictions.
""")

st.success("""
This system demonstrates how data science can transform complex datasets into meaningful insights,
helping bridge the gap between raw data and informed decision-making.
""")