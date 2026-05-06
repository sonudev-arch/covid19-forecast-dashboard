import streamlit as st
import pandas as pd
tabs=st.tabs(["Overview",
            "Holt-Winters",
              "ARIMA",
              "SVR",
              "Fourier Wave",
              "Insights"])
with tabs[0]:
    st.title("Models")
    st.markdown("---")
    st.markdown("## Overview")

    st.write("""
    ### What are Forecasting Models?

    Forecasting models are mathematical and statistical techniques used to predict future values
    based on historical data. In time-series analysis, these models learn patterns such as trends,
    seasonality, and cycles to generate future projections.

    In the context of COVID-19, forecasting models help estimate:
    - Future case counts  
    - Potential wave peaks  
    - Trend continuation or decline  
    """)

    st.markdown("### Models Used")

    st.write("""
    This system implements multiple forecasting models, each capturing different aspects of the data:

    1. **ARIMA (AutoRegressive Integrated Moving Average)**
    - A statistical model based on past values and errors  
    - Works well for stationary time series  
    - Suitable for short-term forecasting  

    2. **Holt-Winters (Exponential Smoothing)**
    - Captures both trend and seasonality  
    - Effective for smooth and structured data patterns  
    - Commonly used in time-series forecasting  

    3. **Support Vector Regression (SVR)**
    - A machine learning model adapted for regression tasks  
    - Handles non-linear relationships in the data  
    - Uses lag-based features for prediction  

    4. **Fourier Model**
    - Uses sinusoidal functions to model periodic behavior  
    - Particularly effective for capturing wave-like patterns  
    - Suitable for modeling repeated COVID-19 waves  
    """)

    st.markdown("### Model Evaluation Approach")

    st.write("""
    To ensure reliable performance, models are evaluated using:

    - **Walk-forward validation**  
    Instead of a single train-test split, the model is repeatedly trained and tested on sequential data,
    mimicking real-world forecasting conditions and avoiding data leakage.

    - **Error Metrics**
    - RMSE (Root Mean Squared Error): Measures magnitude of prediction errors  
    - MAE (Mean Absolute Error): Measures average absolute error  

    These metrics help compare models objectively and identify the most accurate one.
    """)

    st.markdown("### Why Multiple Models?")

    st.write("""
    No single model can capture all characteristics of complex real-world data.
    Each model has strengths and limitations:

    - Statistical models (ARIMA) capture linear dependencies  
    - Smoothing models (Holt-Winters) capture trend and seasonality  
    - Machine learning models (SVR) capture non-linear patterns  
    - Fourier models capture periodic wave behavior  

    By comparing multiple models, this system ensures more robust and reliable forecasting.
    """)

    st.markdown("### Insight")

    st.write("""
    Fourier-based model performs particularly well due to the presence of strong
    wave-like patterns in COVID-19 data. However, final model selection is always validated using error metrics
    such as RMSE and MAE to ensure objective evaluation.
    """)

    st.markdown("---")
with tabs[1]:
    st.markdown("### 1. Holt-Winters Exponential Smoothing")
    st.write("Holt-Winters Exponential Smoothing is a time series forecasting method that accounts for both trend and seasonality in the data. It uses three smoothing equations to capture the level, trend, and seasonal components of the time series. This model is particularly effective for data with strong seasonal patterns.")
    st.write("The model can be expressed as follows:")
    st.latex(r"""
    \begin{aligned}
    l_t &= \alpha y_t + (1 - \alpha)(l_{t-1} + b_{t-1}) \\
    b_t &= \beta (l_t - l_{t-1}) + (1 - \beta) b_{t-1} \\
    s_t &= \gamma \frac{y_t}{l_{t-k} + b_{t-k}} + (1 - \gamma) s_{t-k}
    \end{aligned}
    """)
    st.markdown(" Where:")

    st.markdown(r"""
    - $l_t$: level component at time $t$  
    - $b_t$: trend component at time $t$  
    - $s_t$: seasonal component at time $t$  
    - $\alpha$, $\beta$, $\gamma$: smoothing parameters for level, trend, and seasonality  
    """)
with tabs[2]:
    st.markdown("### 2. ARIMA (AutoRegressive Integrated Moving Average)")
    st.write("ARIMA is a popular time series forecasting model that combines autoregressive (AR), differencing (I), and moving average (MA) components. It is suitable for univariate time series data and can capture various patterns such as trends and seasonality.")
    st.write("The ARIMA model can be expressed as follows:")
    st.latex(r"""
    \begin{aligned}
    \phi(B)(1-B)^d X_t &= \theta(B) \epsilon_t
    \end{aligned}
    """)
    st.write("Where:")

    st.markdown(r"- $\phi(B)$ is the autoregressive polynomial")
    st.markdown(r"- $(1 - B)^d$ is the differencing operator")
    st.markdown(r"- $\theta(B)$ is the moving average polynomial")
    st.markdown(r"- $\epsilon_t$ is the error term")  
with tabs[3]:
    st.markdown("### 3. SVR with lag features(Machine Learning Baseline)")
    st.write("Support Vector Regression (SVR) is a machine learning model that can be used for regression tasks. By creating lag features from the time series data, we can use SVR to capture complex relationships and make predictions. This approach can serve as a baseline for comparing the performance of traditional time series models.")
    st.write("In this approach, we create lag features such as $X_{t-1}$, $X_{t-2}$, etc., and use them as input to the SVR model to predict the target variable at time $t$. The SVR model can capture non-linear relationships in the data, making it a powerful tool for forecasting.")
    st.write("The SVR model can be expressed as follows:")
    st.latex(r"""
    \begin{aligned}
    f(X) &= \sum_{i=1}^{n} \alpha_i K(x_i, x) + b
    \end{aligned}
    """)
    st.write("Where:")
    st.markdown(r"- $f(X)$ is the predicted value")
    st.markdown(r"- $\alpha_i$ are the Lagrange multipliers")
    st.markdown(r"- $K(x_i, x)$ is the kernel function")
    st.markdown(r"- $b$ is the bias term")
with tabs[4]:
    st.markdown('### 4. Fourier Wave model ')
    st.write('The Fourier Wave model is a time series forecasting method that uses Fourier series to capture seasonality in the data. It decomposes the time series into a sum of sine and cosine functions, allowing it to model complex seasonal patterns effectively. This approach is particularly useful when the seasonality is not strictly periodic or when there are multiple seasonalities present in the data.')
    st.write('The Fourier Wave model can be expressed as follows:')
    st.latex(r"""
    \begin{aligned}
    f(t) &= a_0 + \sum_{n=1}^{N} \left( a_n \cos\left(\frac{2\pi n t}{T}\right) + b_n \sin\left(\frac{2\pi n t}{T}\right) \right)
    \end{aligned}
    """)
    st.write("Where:")
    st.markdown(r"- $f(t)$ is the predicted value at time $t$")
    st.markdown(r"- $a_0$ is the DC component")
    st.markdown(r"- $a_n$ and $b_n$ are the Fourier coefficients")
    st.markdown(r"- $T$ is the period of the seasonal pattern") 
    st.markdown('---')
with tabs[5]:       
    st.markdown("## Model Insights")

    st.info("""
    - Holt-Winters captures trend + seasonality but struggles with sudden spikes.

    - ARIMA works well after differencing (d=1), as confirmed by stationarity test.

    - SVR captures non-linear patterns but lacks interpretability.

    - Fourier model performs best due to its ability to capture strong periodic wave patterns observed in decomposition. Final selection should be validated using RMSE and MAE.
    """)
    st.markdown("### Practical Interpretation")

    st.info("""
    - COVID data shows wave-like behavior → Fourier model captures this naturally

    - Sudden spikes (e.g., Wave 2) are difficult for traditional models

    - Hybrid approaches (Fourier + vaccination adjustment) improve realism
    """)
    st.markdown("### Expected Performance")

    df_models = pd.DataFrame({
        "Model": ["Holt-Winters", "ARIMA", "SVR", "Fourier"],
        "Strength": [
            "Simple, interpretable",
            "Good statistical model",
            "Captures non-linearity",
            "Captures wave patterns"
        ],
        "Weakness": [
            "Poor for sudden waves",
            "Needs stationarity",
            "Black-box model",
            "Requires tuning"
        ]
    })
    def highlight_best(row):
        if row["Model"] == "Fourier":
            return ["background-color: #14532d; color: white;"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_models.style.apply(highlight_best, axis=1)
        .set_properties(**{'text-align': 'center'}),
        use_container_width=True
    )
    st.success("""
    Final Conclusion:

    Based on decomposition, stationarity, and observed wave patterns,
    the Fourier-based model (with vaccination adjustment) is the most suitable 
    for long-term COVID forecasting.

    Other models serve as baselines for comparison.
    """)

    st.info(
        "Next, we quantitatively compare these models using RMSE and MAE to identify the best-performing approach."
    )
    st.markdown('##')
    st.markdown('---')