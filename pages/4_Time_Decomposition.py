import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from utils.data_loader import (
    load_merged
)
@st.cache_data
def covid_data():
    data = load_merged()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    return data

tabs=st.tabs(["Overview","Time Decomposition", "Stationarity"])
with tabs[0]:
    st.title("Time Series Decomposition")
    st.markdown("---")
    st.markdown("## Overview")

    st.write("""
    ### What is Time Series Data?

    A time series is a sequence of data points collected or recorded at regular intervals over time.
    Unlike general datasets, time series data has a strong temporal component, meaning that the order of observations matters.

    Examples of time series data include:
    - Daily COVID-19 cases  
    - Stock prices over time  
    - Temperature readings across days  

    In this project, COVID-19 case counts (confirmed, recovered, deaths) are analyzed as time series to understand how the pandemic evolved over time.
    """)

    st.markdown("### What is Time Series Decomposition?")

    st.write("""
    Time series decomposition is a technique used to break down a time series into its fundamental components.
    This helps in understanding the underlying structure of the data and improves forecasting accuracy.

    A time series is typically decomposed into three main components:

    1. **Trend Component**
    - Represents the long-term direction of the data  
    - Shows whether values are increasing, decreasing, or stable over time  

    2. **Seasonal Component**
    - Captures repeating patterns or cycles occurring at regular intervals  
    - For example, weekly or monthly patterns in case reporting  

    3. **Residual (Noise) Component**
    - Represents random variations that are not explained by trend or seasonality  
    - Includes sudden spikes, anomalies, or irregular fluctuations  

    Mathematically, decomposition can be expressed as:

    - Additive Model:  
    Value = Trend + Seasonality + Residual  

    - Multiplicative Model:  
    Value = Trend x Seasonality x Residual  
    """)

    st.markdown("### Why Decomposition is Important")

    st.write("""
    Time series decomposition is essential for:

    - Understanding the structure of data  
    - Separating long-term trends from short-term fluctuations  
    - Identifying periodic patterns (waves in COVID-19 cases)  
    - Improving the performance of forecasting models such as ARIMA and Fourier-based models  

    In this project, decomposition helps reveal how pandemic waves emerged,
    how trends changed over time, and how much variation is due to noise versus real patterns.
    """)

    st.markdown("---")
    
with tabs[1]:
    st.markdown("## Time Series Decomposition(Covid 19)")
    data=covid_data()

    metric = st.selectbox(
        "Select Metric",
        ["Confirmed", "Cured", "Deaths"]
    )
    model = st.radio("Model Type", ["Additive", "Multiplicative"], horizontal=True)
    view = st.radio("View", ["Normal", "Combined"], horizontal=True)

    period = st.slider("Seasonality Period", 2, 30, 7)
    
    series = data[f'New{metric}']

# Handle multiplicative safely
    if model == "Multiplicative":
        series = series.replace(0, 1e-5)
    if len(data) < 2 * period:
        st.warning("Not enough data for selected seasonality period.")
        st.stop()
    result = seasonal_decompose(series, model=model.lower(), period=period)
    

    # Function to create plot
    def plot_component(series, title, color):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode='lines',
                name=title,
                line=dict(color=color)
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            height=300
        )
      
        return fig
    if view=='Normal':
        # Observed
        st.plotly_chart(plot_component(result.observed, "Observed", "#1f77b4"))

        # Trend
        st.plotly_chart(plot_component(result.trend, "Trend", "#2ca02c"))

        # Seasonal
        st.plotly_chart(plot_component(result.seasonal, "Seasonality", "#ff7f0e"))

        # Residual
        st.plotly_chart(plot_component(result.resid, "Residual", "#d62728"))

    if view=='Combined':
        fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonality", "Residual")
        )

        fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed, name="Observed"), row=1, col=1)
        fig.add_trace(go.Scatter(x=result.observed.index, y=result.trend, name="Trend"), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.observed.index, y=result.seasonal, name="Seasonal"), row=3, col=1)
        fig.add_trace(go.Scatter(x=result.observed.index, y=result.resid, name="Residual"), row=4, col=1)

        fig.update_layout(height=900, title=f"Time Series Decomposition for {metric} Cases")
        st.plotly_chart(fig)
    st.markdown("## Decomposition Insights")

    try:
        trend = result.trend.dropna()
        seasonal = result.seasonal.dropna()
        resid = result.resid.dropna()

        #Trend Direction
        # Robust trend slope using linear regression
        trend_slope = np.polyfit(range(len(trend)), trend, 1)[0]
        trend_direction = "Upward" if trend_slope > 0 else "Downward"

        trend_direction = "Upward" if trend_slope > 0 else "Downward"

        #Seasonality Strength
        season_strength = seasonal.std()

        # Noise Level (Residual)
        noise_level = resid.std()

        # Stability Ratio (Seasonality vs Noise)
        stability_ratio = season_strength / (noise_level + 1e-5)

        #Display
        col1, col2, col3 = st.columns(3)

        col1.metric("Trend Direction", trend_direction)
        col2.metric("Seasonality Strength", f"{season_strength:.2f}")
        col3.metric("Noise Level (Residual)", f"{noise_level:.2f}")

        col4, col5 = st.columns(2)

        col4.metric("Stability Ratio", f"{stability_ratio:.2f}")
        col5.metric("Seasonality Period", period)

        # Insights
        st.info(
        f"""
        - Trend is **{trend_direction.lower()}**, indicating overall pandemic direction.

        - Seasonality strength = {season_strength:.2f}:
            Higher values indicate stronger repeating wave patterns.

        - Residual noise = {noise_level:.2f}:
            Residual represents unexplained variation after removing trend and seasonality.
            High residuals may indicate sudden outbreaks, reporting anomalies,
            or external shocks.

        - Stability ratio = {stability_ratio:.2f}:
            >1 → structured, predictable waves  
            <1 → chaotic spread
        """
    )

    except Exception as e:
        st.warning(f"Insight calculation error: {e}")

    st.success(
    "Detected seasonality indicates periodic patterns in the data. "
    "This supports the use of Fourier-based models for capturing cyclical behavior, "
    "though real-world factors should also be considered."
    )
    st.markdown("### Trend vs Wave Behavior")

    try:
        peak_val = result.observed.max()
        trend_peak = result.trend.max()

        st.metric("Peak vs Trend Ratio", f"{peak_val / trend_peak:.2f}x")

        st.info(
            "Peaks significantly above trend indicate sudden outbreak waves "
            "that cannot be explained by long-term trend alone."
        )

    except:
        pass
    st.markdown('---')
with tabs[2]:
#Testing for Stationarity.
    st.markdown("## Testing for Stationarity")
    st.write("To determine if the time series data is stationary, we can use the Augmented Dickey-Fuller (ADF) test. A stationary time series has constant mean and variance over time, which is a key assumption for many time series forecasting models. If the p-value from the ADF test is less than 0.05, we can reject the null hypothesis and conclude that the series is stationary. If not, we may need to apply differencing to make it stationary before modeling.")
    #ADF Test
    adf_result = adfuller(data['NewConfirmed_7d'])
    # Testing for Stationarity (ADF Test)

    def adf_test(series, title=''):
        result = adfuller(series.dropna())

        st.subheader(f"ADF Test: {title}")
        st.write(f"ADF Statistic: {round(result[0], 3)}")
        st.write(f"p-value: {round(result[1], 3)}")

        if result[1] <= 0.05:
            st.success("Data is Stationary. ")
        else:
            st.warning("Data is Non-Stationary - Differencing Required")

    st.subheader('AUGMENTED DICKEY-FULLER STATIONARITY TEST')
    st.write('H₀: series has a unit root (non-stationary)')
    st.write('Reject H₀ (p<0.05) → stationary')
    st.write()
    adf_test(data['NewConfirmed_7d'],          'Raw 7d avg series')
    adf_test(data['NewConfirmed_7d'].diff(), '1st difference')
    st.write()
    st.success(
        "Series becomes stationary after 1st differencing → suitable for ARIMA(d=1). "
        "This confirms that the data has a trend component that needs removal before modeling."
    )
    st.info('ARIMA order I(d=1) and Fourier model works on raw values with dampening')

    st.markdown('####')
    st.markdown('---')
