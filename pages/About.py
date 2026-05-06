import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.title("About")
st.caption("Understanding the motivation, methodology, and analytical depth of the COVID-19 Analytics System")

st.markdown("---")

st.markdown("## Purpose")

st.write("""
The COVID-19 pandemic created a highly dynamic and complex data environment, where understanding trends,
predicting future behavior, and evaluating interventions became critical.

This project was developed to address these challenges by building a comprehensive analytical system
that leverages time-series analysis, statistical modeling, and machine learning.

The primary objective is to transform raw pandemic data into:
- Meaningful insights
- Interpretable patterns
- Reliable forecasts

The system is designed not just for visualization, but for deep analytical understanding and decision support.
""")

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

1. Data Processing
- Data cleaning and preprocessing  
- Handling missing values and inconsistencies  
- Time-series structuring  

2. Exploratory Data Analysis (EDA)
- Trend visualization  
- Peak detection and wave identification  
- Comparative analysis across regions  

3. Feature Engineering
- Active cases computation  
- Growth rate (percentage change)  
- Moving averages and smoothing  

4. Time-Series Decomposition
- Separation into Trend, Seasonality, and Residual  
- Identification of underlying patterns  

5. Statistical Testing
- Stationarity testing using Augmented Dickey-Fuller (ADF)  
- Differencing to stabilize time-series  

6. Model Development
The following models are implemented:

- ARIMA (statistical forecasting)  
- Holt-Winters (trend and seasonality smoothing)  
- Support Vector Regression (SVR)  
- Fourier-based modeling for periodic patterns  

7. Model Evaluation
- Walk-forward validation  
- Error metrics: RMSE and MAE  

8. Forecasting and Wave Prediction
- Future trend projection  
- Peak detection on predicted data  
- Vaccination-adjusted forecasting  
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
This project was developed as part of a learning journey in data science and analytics.

It reflects a strong focus on:
- Understanding real-world datasets  
- Applying theoretical concepts in practical scenarios  
- Building scalable and interactive analytical systems  

The project demonstrates skills in:
- Data analysis and visualization  
- Time-series modeling and forecasting  
- Machine learning fundamentals  
- Streamlit-based application development  
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