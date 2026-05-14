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
    This section compares multiple forecasting models applied to COVID-19 time-series data.

    Each model captures different characteristics such as trend, seasonality, non-linearity, and wave patterns.
    The goal is to evaluate their behavior and identify the most reliable model for forecasting.
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

    st.write("""
    Holt-Winters is a time-series forecasting method that captures level, trend, and seasonality
    using exponential smoothing.

    It is particularly effective for data with consistent patterns and gradual changes,
    as it smooths noise and emphasizes underlying structure.

    In the context of COVID-19 data:
    - It models overall trend and seasonal behavior effectively  
    - It performs well when the data follows smooth and predictable patterns  
    - However, it struggles with sudden spikes and irregular wave patterns  

    Since COVID-19 data contains abrupt changes (e.g., sudden waves),
    Holt-Winters may fail to capture sharp peaks accurately.

    Thus, it is used as a baseline model for trend and seasonality comparison,
    rather than for capturing complex wave dynamics.
    """)
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

    st.write("""
    ARIMA is a statistical time-series model that captures relationships between past values
    and past errors using autoregression, differencing, and moving average components.

    It is particularly effective for modeling linear patterns in stationary data.

    In this project:
    - Differencing (d=1) is applied to make the data stationary, as confirmed by ADF testing  
    - ARIMA captures short-term dependencies and local trends effectively  
    - It performs well for short-term forecasting  

    However:
    - It assumes linear relationships and struggles with non-linear patterns  
    - It cannot naturally capture strong wave-like or periodic behavior  
    - Performance degrades in the presence of sudden spikes and structural changes  

    Thus, ARIMA is used as a strong statistical baseline for short-term forecasting,
    but is not sufficient for capturing long-term wave dynamics in COVID-19 data.
    """)
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
    st.markdown("### 3. SVR with Lag Features (Machine Learning Model)")

    st.write("""
    Support Vector Regression (SVR) is a machine learning model used for regression tasks that can capture
    non-linear relationships in time-series data.

    In this approach, lag features are created (e.g., $X_{t-1}$, $X_{t-2}$, ...) to convert the time-series problem
    into a supervised learning problem.

    SVR uses kernel functions to map input data into a higher-dimensional space,
    allowing it to model complex patterns that traditional statistical models may miss.

    In the context of COVID-19 data:
    - SVR performs well for short-term predictions  
    - It captures non-linear fluctuations and local variations  
    - However, it lacks interpretability and struggles with long-term trends and structural changes  

    Thus, SVR is used as a complementary model for capturing short-term non-linear behavior,
    rather than as the primary forecasting model.
    """)
  
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
    
    st.markdown("### 4. Fourier Wave Model")

    st.write("""
    The Fourier Wave model is a time-series forecasting approach that represents data
    as a combination of sine and cosine functions, allowing it to capture periodic patterns.

    It is particularly effective for data that exhibits wave-like or cyclic behavior.

    In the context of COVID-19 data:
    - The pandemic shows repeated waves driven by transmission cycles and external factors  
    - Fourier decomposition captures these periodic patterns effectively  
    - It produces smooth approximations of the underlying trend  

    This makes the model especially useful for:
    - Long-term forecasting  
    - Identifying recurring wave structures  
    - Supporting stable peak detection by reducing noise  

    However:
    - The model requires careful tuning of frequency components  
    - It may not capture sudden irregular spikes perfectly  

    Thus, the Fourier model is the most suitable approach for modeling wave-like behavior
    and is used as the primary model for long-term forecasting in this project.
    """)
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
    st.markdown("### Model Selection Insight")

    st.info("""
    - Holt-Winters captures trend and seasonality but struggles with sudden spikes and irregular wave patterns.

    - ARIMA performs well after differencing but assumes linear relationships and short-term dependencies.

    - SVR captures non-linear patterns using lag features, but lacks interpretability and struggles with long-term structure.

    - Fourier model is particularly effective because COVID-19 data exhibits strong wave-like and periodic behavior.

    By decomposing the signal into sinusoidal components, the Fourier model captures underlying cycles
    that traditional models fail to represent, making it more suitable for long-term forecasting.

    - Fourier also enables smooth reconstruction of trends, which makes peak detection more stable and reliable
    compared to noisy daily data.
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
    st.markdown("### Key Takeaway")
    st.info("Model selection and performance depend on how well the model aligns with the underlying data characteristics.")
    st.success("""
    Final Conclusion:

    COVID-19 data exhibits strong wave-like and non-stationary behavior.

    The Fourier-based model performs best for long-term forecasting because it captures periodic patterns,
    while traditional models struggle with sudden spikes and structural changes.

    Other models serve as baselines for comparison and short-term behavior analysis.
    """)

    st.info(
        "Next, we quantitatively compare these models using RMSE and MAE to identify the best-performing approach."
    )
    st.markdown("### Model Comparison Strategy")

    st.write("""
    All models are evaluated using the same dataset and validation strategy.

    Performance is compared using:
    - RMSE (penalizes large errors)
    - MAE (average prediction error)

    This ensures fair and objective comparison across models.
    """)
    st.markdown("### Final Insight")
    st.success("Understanding the data is more important than choosing complex models.")
    st.markdown('---')