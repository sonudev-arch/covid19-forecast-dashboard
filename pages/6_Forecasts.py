#Importing necessary libraries and modules
from datetime import timedelta
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge
from utils.holt_winters import HoltWinters
from utils.arima_model import SimpleARIMA
from sklearn.preprocessing import MinMaxScaler
from utils.data_loader import load_merged
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
import sys
import json
import joblib
import pandas as pd
import numpy as np

#Cache functions for data loading and model evaluation

# Data loading
@st.cache_data
def load_data_cached():
    df = load_merged()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")


# Holt-Winters Model
@st.cache_resource
def load_hw_cached():
    model = joblib.load("models/holt_winters_model.joblib")
    meta = joblib.load("models/holt_winters_meta.joblib")
    with open("models/holt_winters_params.json") as f:
        state = json.load(f)
    return model, meta, state


@st.cache_data
def hw_walk_forward_cached(train, val):
    history = list(train)
    preds = []
    for actual in val:
        model = HoltWinters().fit(np.array(history))
        pred = model.predict(1)
        yhat = pred[0] if isinstance(pred, (list, np.ndarray)) else float(pred)
        preds.append(max(yhat, 0))
        history.append(actual)
    return np.array(preds)


# ARIMA Model
@st.cache_resource
def load_arima_cached():
    model = joblib.load("models/arima_model.joblib")
    with open("models/arima_params.json") as f:
        params = json.load(f)
    return model, params


def arima_walk_forward(model, train, val):
    #  Ensure numpy arrays
    train = np.asarray(train, dtype=float)
    val   = np.asarray(val, dtype=float)

    #  Force integer parameters
    p = int(model.p)
    q = int(model.q)

    #  Initialize differenced history
    dh = list(np.diff(train))

    # Safe residual initialization
    if hasattr(model, "_res") and len(model._res) >= q:
        rh = list(np.asarray(model._res).flatten())
    else:
        rh = [0.0] * max(q, 1)

    last_y = train[-1]
    preds = []

    for actual in val:

        #  Safe slicing
        ar = dh[-p:] if len(dh) >= p else [0.0]*p
        ma = rh[-q:] if len(rh) >= q else [0.0]*q

        ar = np.array(ar)[::-1]
        ma = np.array(ma)[::-1]

        #  Predict
        try:
            dp = model.predict(np.concatenate([ar, ma]).reshape(1, -1))[0]
        except:
            dp = 0.0  # fallback safety

        yhat = max(last_y + dp, 0)
        preds.append(yhat)

        # Update history
        new_diff = actual - last_y
        dh.append(new_diff)
        rh.append(0.0)

        last_y = actual

    return np.array(preds)

@st.cache_data
def arima_walk_forward_cached(_model, train_series, val_series):
    return arima_walk_forward(_model, train_series, val_series)


# Support Vector Regression (SVR)
@st.cache_resource
def load_svr_cached():
    svr = joblib.load("models/svr_model.joblib")
    with open("models/svr_state.json") as f:
        state = json.load(f)
    return svr, state


@st.cache_data
def compute_svr_eval_cached(train_series, val_series, history, _scaler, _svr, LAGS):
    y_all = np.concatenate([train_series, val_series]).astype(float)
    y_sc  = _scaler.transform(y_all.reshape(-1,1)).flatten()

    hist = list(history.copy())
    n_lags = len(LAGS)
    preds_scaled = []

    for i in range(len(val_series)):
        X = np.array(hist[-n_lags:]).reshape(1, -1)
        p = _svr.predict(X)[0]
        preds_scaled.append(p)

        actual_scaled = y_sc[len(train_series) + i]
        hist.append(actual_scaled)

    preds = _scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1,1)
    ).flatten()

    return np.maximum(preds, 0)

# Fourier Wave Model
@st.cache_resource
def load_fourier_cached():
    meta = joblib.load("models/fourier_vanilla_forecast.joblib")
    with open("models/fourier_model_params.json") as f:
        state = json.load(f)
    return meta, state


@st.cache_data
def fourier_eval_cached(train_len, val_len, params):
    t = np.arange(train_len, train_len + val_len, dtype=float)
    preds = np.clip(wave_model(t, *params), 0, None)
    return preds


# Fourier with Vaccination Adjustment
@st.cache_data
def load_vacc_cached():
    with open("models/fourier_vacc_model.json") as f:
        state = json.load(f)

    dates = pd.to_datetime(state["dates"])

    return (
        pd.Series(state["fourier_vacc_adj"], index=dates),
        pd.Series(state["fourier_vanilla"], index=dates),
        pd.Series(state["suppression"], index=dates),
        state["lag_days"],
        state["decay"],
        state["vacc_sensitivity"]
    )
@st.cache_resource
def load_all_models():
    return {
        "hw": load_hw_cached(),
        "arima": load_arima_cached(),
        "svr": load_svr_cached(),
        "fourier": load_fourier_cached()
    }
#Classes
sys.modules['__main__'].HoltWinters = HoltWinters
sys.modules['__main__'].SimpleARIMA = SimpleARIMA

POP_INDIA  = 1_380_000_000          
TRAIN_CUTOFF = '2021-07-01'         
FORECAST_DAYS = 365 * 9 

#Peak Prediction function

def detect_peaks_scipy(series, height=None, distance=60, prominence=None):
    """
    Detect peaks using scipy.

    Parameters:
    - series: pandas Series (must have datetime index)
    - height: minimum peak height
    - distance: minimum distance between peaks (in days)
    - prominence: peak prominence (controls sharpness)

    Returns:
    - list of tuples: (index, date, value)
    """

    values = series.values

    peaks, properties = find_peaks(
        values,
        height=height,
        distance=distance,
        prominence=prominence
    )

    results = [
        (i, series.index[i], values[i])
        for i in peaks
    ]

    return results
#Main Merged Dataset
merged = load_data_cached()
merged["Date"] = pd.to_datetime(merged["Date"])

series = merged["NewConfirmed_7d"].copy()
series.index = merged["Date"]

TRAIN_CUTOFF = pd.to_datetime(TRAIN_CUTOFF)
train_len    = series[series.index <  TRAIN_CUTOFF]
val_len      = series[series.index >= TRAIN_CUTOFF]
merged = merged.set_index("Date")  

#Loading Data
#Holt-Winters Model Data
hw_model, hw_meta, hw_state = load_hw_cached()
series_len = hw_meta["series_len"]
hw_last_date  = pd.Timestamp(hw_meta["last_date"])
hw_residual_std = hw_meta.get("residual_std", None)
#ARIMA Model
arima_model, arima_params = load_arima_cached()
p = arima_params["p"]
q = arima_params["q"]
alpha = arima_params["alpha"]
arima_last_date = pd.to_datetime(arima_params["last_date"])
#SVR Model
svr, svr_state = load_svr_cached()
LAGS = svr_state["lags"]
svr_history = np.array(svr_state["history"])

# Rebuild scaler
scaler = MinMaxScaler()
scaler.feature_range = (0, 1)
scaler.min_ = np.array(svr_state["scaler_min"])
scaler.scale_ = np.array(svr_state["scaler_scale"])
scaler.data_min_ = np.array(svr_state["data_min"])
scaler.data_max_ = np.array(svr_state["data_max"])
scaler.n_features_in_ = 1
#Fourier Wave Model Data
fourier_meta, fourier_state = load_fourier_cached()

try:
    params = np.array([
        fourier_state["A1"], fourier_state["ω1"], fourier_state["φ1"],
        fourier_state["A2"], fourier_state["ω2"], fourier_state["φ2"],
        fourier_state["A3"], fourier_state["ω3"], fourier_state["φ3"],
        fourier_state["C"], fourier_state["decay"]
    ])
except KeyError as e:
    raise KeyError(f"Missing parameter in JSON: {e}")

# Validate parameter count
assert len(params) == 11, "Expected 11 parameters for wave_model"

# Metadata from joblib
fourier_series_len = fourier_meta.get("series_len")
fourier_last_date = fourier_meta.get("last_date")

# Fallback to JSON if missing
if fourier_series_len is None:
    fourier_series_len = fourier_state.get("series_len")

if fourier_last_date is None:
    fourier_last_date = fourier_state.get("last_date")

# Final validation
if fourier_series_len is None:
    raise ValueError("series_len missing in BOTH joblib and JSON")

if fourier_last_date is None:
    raise ValueError("last_date missing in BOTH joblib and JSON")

fourier_last_date = pd.Timestamp(fourier_last_date)

# Optional (for confidence interval)
fourier_residual_std = fourier_meta.get("residual_std", None)

def wave_model(t, A1, omega1, phi1,
                  A2, omega2, phi2,
                  A3, omega3, phi3,
                  C, decay):
    damp = np.exp(-decay * t)
    return (C * damp
            + A1 * np.sin(omega1*t + phi1) * damp
            + A2 * np.sin(omega2*t + phi2) * damp
            + A3 * np.sin(omega3*t + phi3) * np.exp(-decay*0.5*t))

# Fourier Adjusted Model
fourier_vacc_adj, fourier_vanilla, suppression_forecast, LAG_DAYS, DECAY, VACC_SENSITIVITY = load_vacc_cached()
st.markdown('<a id="top"></a>', unsafe_allow_html=True)
st.markdown("######")
tabs = st.tabs([
    "Overview",
    "Forecast",
    "Holt-Winters",
    "ARIMA",
    "SVR",
    "Fourier",
    "Model Comparison",
    "Summary"
])
with tabs[0]:
    st.title("Forecasts")
    st.markdown("---")
    st.markdown("## Overview")
    st.write("""
    This section generates and evaluates forecasts for COVID-19 time-series data
    using multiple models.

    The goal is to compare model performance, analyze future trends,
    and identify potential wave patterns using data-driven methods.
    """)
    st.markdown("### Forecasting Approach")

    st.write("""
    This system applies multiple forecasting techniques to generate reliable predictions.
    Each model captures different characteristics of the time series, and their outputs are compared
    to ensure robustness and accuracy.

    The forecasting pipeline includes:
    - Model training on historical data  
    - Walk-forward validation for realistic evaluation  
    - Future projection based on learned patterns  
    - Peak detection on predicted values to identify upcoming waves  
        """)
    st.markdown("### Overfitting Control")

    st.info("""
    Walk-forward validation is used to reduce overfitting.

    Models are evaluated on unseen future data sequentially,
    ensuring that predictions mimic real-world forecasting conditions.
    """)
    st.markdown("### Data Leakage Prevention")

    st.info("""
    Data leakage is avoided by ensuring that models are trained only on past data
    and evaluated on future unseen values.

    Walk-forward validation enforces strict temporal separation,
    making the evaluation realistic and reliable.
    """)
    st.markdown("### Types of Forecasts Generated")

    st.write("""
The system generates different types of forecasts:

1. **Full Timeline Forecast**
   - Combines historical data with future projections  
   - Provides a continuous view from past to predicted future  

2. **Model-Specific Forecasts**
   - Individual forecasts from ARIMA, Holt-Winters, SVR, and Fourier models  
   - Helps understand how each model behaves  

3. **Adjusted Forecast (Vaccination Impact)**
   - Incorporates real-world vaccination effects  
   - Adjusts predicted case counts based on assumed suppression  

4. **Wave Forecast**
   - Uses peak detection on predicted data  
   - Identifies potential future waves and their severity  
""")

    st.markdown("### Uncertainty and Confidence")

    st.write("""
Forecasting involves uncertainty, as future values depend on many unknown factors.
To address this, the system includes confidence ranges around predictions.

These ranges indicate:
- Possible variation in predictions  
- Degree of uncertainty in model outputs  

This helps in interpreting forecasts more realistically rather than as exact values.
""")

    st.markdown("### Why Forecasting is Important")

    st.write("""
Forecasting provides insights into future scenarios, helping in:

- Anticipating potential outbreaks or waves  
- Understanding long-term trends  
- Evaluating the impact of interventions such as vaccination  
- Supporting data-driven decision making  

Forecasting is not only used for prediction,
but also for understanding the behavior and dynamics of pandemic waves.
""")

    cols = st.columns(6)
    with cols[0]:
        st.markdown('<a href="#top">Go to Top</a>', unsafe_allow_html=True)

@st.cache_data
def get_hist_peaks(series):
    return detect_peaks_scipy(series, height=50000, distance=60, prominence=10000)

hist_peaks_info = get_hist_peaks(series)
@st.cache_data
def get_future_peaks_vacc(series):
    return detect_peaks_scipy(
    series,
    height=3000,
    distance=90,
    prominence=1000
)

future_peaks_vacc=get_future_peaks_vacc(fourier_vacc_adj)
if future_peaks_vacc:
    next_wave_date = future_peaks_vacc[0][1]
    next_wave_cases = future_peaks_vacc[0][2]
else:
    next_wave_date = "N/A"
    next_wave_cases = 0
@st.cache_data
def get_future_peaks_van (series):
    return detect_peaks_scipy(
    series,
    height=5000,
    distance=90,
    prominence=1500
)
future_peaks_van=get_future_peaks_van(fourier_vanilla)

    # Historical
hist_df = pd.DataFrame([
        {"Wave": i+1, "Date": d.date(), "Cases/day": int(v)}
        for i, (_, d, v) in enumerate(hist_peaks_info)
    ])

    # Future (Vaccination Adjusted)
start_wave = len(hist_peaks_info) + 1

future_df = pd.DataFrame([
        {"Wave": start_wave + i, "Date": d.date(), "Cases/day": int(v)}
        for i, (_, d, v) in enumerate(future_peaks_vacc[:10])
    ])

with tabs[1]:
    st.markdown("## Forecast Detail (2020 to 2030)")


    fig1 = go.Figure()

    # Historical
    fig1.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        name="Historical",
        line=dict(width=2),
        fill='tozeroy',
        opacity=0.3
    ))

    # Fourier Vanilla
    fig1.add_trace(go.Scatter(
        x=fourier_vanilla.index,
        y=fourier_vanilla.values,
        name="Fourier Vanilla",
        line=dict(dash='dash', width=2),
        opacity=0.6
    ))

    # Vaccination Adjusted
    fig1.add_trace(go.Scatter(
        x=fourier_vacc_adj.index,
        y=fourier_vacc_adj.values,
        name="Fourier + Vaccination Adjusted",
        line=dict(width=3)
    ))

    # Confidence band (±30%)
    upper = fourier_vacc_adj * 1.3
    lower = fourier_vacc_adj * 0.7

    fig1.add_trace(go.Scatter(x=fourier_vacc_adj.index, y=upper,
                            line=dict(width=0), showlegend=False))

    fig1.add_trace(go.Scatter(
        x=fourier_vacc_adj.index,
        y=lower,
        fill='tonexty',
        name="±30% Band",
        opacity=0.2,
        line=dict(width=0)
    ))

    # Historical peak labels
    fig1.add_trace(go.Scatter(
        x=[p[1] for p in hist_peaks_info],
        y=[p[2] for p in hist_peaks_info],
        mode="markers+text",
        text=[f"W{i+1}" for i in range(len(hist_peaks_info))],
        textposition="top center",
        name="Historical Peaks"
    ))

    # Future peak labels
    fig1.add_trace(go.Scatter(
        x=[p[1] for p in future_peaks_vacc[:8]],
        y=[p[2] for p in future_peaks_vacc[:8]],
        mode="markers+text",
        text=[f"W{len(hist_peaks_info)+1+i}" for i in range(len(future_peaks_vacc[:8]))],
        textposition="top center",
        name="Predicted Peaks"
    ))

    fig1.update_layout(
        title="Full Timeline 2020 to 2030 (Fourier Models)",
        xaxis_title="Year",
        yaxis_title="Cases/day",
        hovermode="x unified",
        template="plotly_dark"
    )

    st.plotly_chart(fig1, width="stretch")

    st.markdown("###  Forecast Detail (Aug 2021 to 2030)")

    fig2 = go.Figure()

    # Main forecast
    fig2.add_trace(go.Scatter(
        x=fourier_vacc_adj.index,
        y=fourier_vacc_adj.values,
        name="Vaccination Adjusted",
        line=dict(width=3)
    ))

    # ±30% band
    fig2.add_trace(go.Scatter(
        x=fourier_vacc_adj.index,
        y=upper,
        line=dict(width=0),
        showlegend=False
    ))

    fig2.add_trace(go.Scatter(
        x=fourier_vacc_adj.index,
        y=lower,
        fill='tonexty',
        name="±30% uncertainty",
        opacity=0.25,
        line=dict(width=0)
    ))

    # Peak annotations (detailed)
    w_num = len(hist_peaks_info) + 1

    for i, (_, date, val) in enumerate(future_peaks_vacc[:12]):
        fig2.add_annotation(
            x=date,
            y=val,
            text=f"Wave {w_num+i}<br>{date.strftime('%b %Y')}<br>{val/1e3:.1f}K/day",
            showarrow=True,
            arrowhead=2,
            yshift=20,
            font=dict(size=10)
        )

    fig2.update_layout(
        title="Forecast Detail: Aug 2021 to Dec 2030",
        xaxis_title="Year",
        yaxis_title="Daily Cases",
        xaxis_range=[pd.Timestamp("2021-08-01"), pd.Timestamp("2030-12-31")],
        hovermode="x unified",
        template="plotly_dark"
    )

    st.plotly_chart(fig2, width="stretch")

    #Adjusted Forecast with Vaccination Suppression
    st.markdown("## Fourier Model with Vaccination Adjustment")

    if "vacc_days" not in st.session_state:
        st.session_state.vacc_days = 30

    def _sync_slider_fourier_vacc():
        st.session_state.vacc_days = st.session_state.vacc_slider

    def _sync_input_fourier_vacc():
        st.session_state.vacc_days = st.session_state.vacc_input

    st.slider(
        "Select Forecast Days",
        30, len(fourier_vacc_adj),
        value=st.session_state.vacc_days,
        key="vacc_slider",
        on_change=_sync_slider_fourier_vacc,
    )

    st.number_input(
        "Enter Forecast Days",
        min_value=30, max_value=len(fourier_vacc_adj),
        value=st.session_state.vacc_days,
        key="vacc_input",
        on_change=_sync_input_fourier_vacc,
    )

    vacc_days = st.session_state.vacc_days

    if st.button("Generate Forecast", key="vacc_button"):

        # Slice data
        vacc_series = fourier_vacc_adj.iloc[:vacc_days]
        vanilla_series = fourier_vanilla.iloc[:vacc_days]
        suppression_series = suppression_forecast.iloc[:vacc_days]

    
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=vanilla_series.index,
            y=vanilla_series,
            name="Vanilla Fourier"
        ))

        fig.add_trace(go.Scatter(
            x=vacc_series.index,
            y=vacc_series,
            name="Vaccination Adjusted"
        ))

        fig.add_trace(go.Scatter(
            x=suppression_series.index,
            y=suppression_series,
            name="Suppression",
            yaxis="y2"
        ))

        fig.update_layout(
            title="Fourier Forecast with Vaccination Adjustment",
            xaxis_title="Date",
            yaxis=dict(title="Cases"),
            yaxis2=dict(title="Suppression", overlaying='y', side='right'),
            template="plotly_dark",
            hovermode="x unified"
        )

        st.plotly_chart(fig, width="stretch")

    
        vacc_series_clean = (
        vacc_series
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0)
    )
        df = vacc_series_clean.round().astype(int).reset_index()
        df.columns = ["Date", "Predicted Cases"]
        df["Date"] = df["Date"].dt.date
        df = df.set_index("Date")

        st.dataframe(df)
        st.download_button(
        "Download Prediction CSV",
        df.to_csv(index=False).encode("utf-8"),
        "fourier_ad_pred.csv",
        "text/csv"
    )
    cols = st.columns(6)
    with cols[0]:
        st.markdown('<a href="#top">Go to Top</a>', unsafe_allow_html=True)

with tabs[2]:
    st.markdown("## Holt-Winters Forecast")

    if "hw_days" not in st.session_state:
        st.session_state.hw_days = 30

    def _sync_slider_hw():
        st.session_state.hw_days = st.session_state.hw_slider

    def _sync_input_hw():
        st.session_state.hw_days = st.session_state.hw_input

    st.slider(
        "Select Forecast Days",
        30, 3285,
        value=st.session_state.hw_days,
        key="hw_slider",
        on_change=_sync_slider_hw,
    )

    st.number_input(
        "Enter Forecast Days",
        min_value=30, max_value=3285, step=30,
        value=st.session_state.hw_days,
        key="hw_input",
        on_change=_sync_input_hw,
    )

    hw_days = st.session_state.hw_days



    if st.button("Generate Forecast", key="hw_button"):

        forecast = hw_model.predict(hw_days)

        hw_dates = pd.date_range(
            hw_last_date + pd.Timedelta(days=1),
            periods=hw_days,
            freq="D"
        )

        forecast_series = pd.Series(forecast, index=hw_dates).round()

        st.line_chart(forecast_series)

        df = forecast_series.astype(int).reset_index()
        df.columns = ["Date", "Predicted Cases"]
        df["Date"] = df["Date"].dt.date
        df = df.set_index("Date")

        st.dataframe(df)
        st.download_button(
        "Download Prediction CSV",
        df.to_csv(index=False).encode("utf-8"),
        "hw_pred.csv",
        "text/csv"
    )
    st.write(f"Season Length: {hw_state['season_length']}")

    st.write(f"Series Length: {hw_state['series_len']}")
    st.write(f"Last Date: {hw_state['last_date']}")

    def hw_walk_forward(train, val):

        history = list(train)
        preds = []

        for actual in val:
            # Re-fit model on updated history
            model = HoltWinters().fit(np.array(history))

            pred = model.predict(1)
            yhat = pred[0] if isinstance(pred, (list, np.ndarray)) else float(pred)
            preds.append(max(yhat, 0)) 
            # update with actual value
            history.append(actual)

        return np.array(preds)

    st.markdown("### Holt-Winters Validation and Evaluation")

    try:
        with st.spinner("Running Holt-Winters validation..."):
            hw_preds = hw_walk_forward_cached(train_len,val_len)

        hw_rmse = np.sqrt(mean_squared_error(val_len, hw_preds))
        hw_mae  = mean_absolute_error(val_len, hw_preds)

        compare_df = pd.DataFrame({
            "Actual": val_len,
            "Predicted": hw_preds
        })

        hw_val_dates = val_len.index if hasattr(val_len, "index") else range(len(val_len))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hw_val_dates,
            y=val_len,
            mode='lines',
            name='Actual'
        ))

        fig.add_trace(go.Scatter(
            x=hw_val_dates,
            y=hw_preds,
            mode='lines',
            name='Predicted'
        ))

        fig.update_layout(
            title="Holt-Winters Validation: Actual vs Predicted",
            xaxis_title="Date",
            yaxis_title="Cases",
            template="plotly_dark",
            hovermode="x unified"
        )

        st.plotly_chart(fig, width="stretch")

    except Exception as e:
        st.error(f"Holt-Winters Error: {e}")


    #Forecast range
    future_dates = pd.date_range(
        hw_last_date + pd.Timedelta(days=1),
        periods=365,
        freq='D'
    )

    st.write(f'Forecast range: {future_dates[0].date()} → {future_dates[-1].date()}')
    cols = st.columns(6)
    with cols[0]:
        st.markdown('<a href="#top">Go to Top</a>', unsafe_allow_html=True)

#ARIMA Model
with tabs[3]:
    st.markdown("## ARIMA Model")

    st.write(f"Last Training Date: {arima_last_date.date()}")
    st.subheader("Forecast")
    if "arima_days" not in st.session_state:
        st.session_state.arima_days = 30

    def _sync_slider_arima():
        st.session_state.arima_days = st.session_state.arima_slider

    def _sync_input_arima():
        st.session_state.arima_days = st.session_state.arima_input

    st.slider(
        "Select Forecast Days",
        30, 3285,
        value=st.session_state.arima_days,
        key="arima_slider",
        on_change=_sync_slider_arima,
    )
    st.number_input(
        "Enter Forecast Days",
        min_value=30, max_value=3285, step=30,
        value=st.session_state.arima_days,
        key="arima_input",
        on_change=_sync_input_arima,
    )
    arima_days = st.session_state.arima_days
    if st.button("Generate Forecast", key="arima_button"):

        arima_preds = np.asarray(arima_model.predict(arima_days)).flatten()

        arima_dates = pd.date_range(
            start=arima_last_date + timedelta(days=1),
            periods=arima_days,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            "Date": arima_dates,
            "Predicted Cases": arima_preds.astype(int)
        }).set_index("Date")

        st.line_chart(forecast_df)
        st.dataframe(forecast_df)
        st.download_button(
        "Download Prediction CSV",
        forecast_df.to_csv(index=False).encode("utf-8"),
        "arima_pred.csv",
        "text/csv"
    )
    def predict_one_step(model, dh, rh):
        p = int(model.p)
        q = int(model.q)

        ar = np.array(dh[-p:] if len(dh) >= p else [0.0]*p)[::-1]
        ma = np.array(rh[-q:] if len(rh) >= q else [0.0]*q)[::-1]

        return model.predict(np.concatenate([ar, ma]).reshape(1, -1))[0]

    st.subheader(" ARIMA Validation and Evaluation")

    try:
        val_alen=np.asarray(val_len)
        train_alen=np.asarray(train_len)
    
        arima_preds = arima_walk_forward_cached(arima_model, train_alen, val_alen)
    
        arima_rmse = np.sqrt(mean_squared_error(val_alen, arima_preds))
        arima_mae = mean_absolute_error(val_alen, arima_preds)

        compare_df = pd.DataFrame({
            "Actual": val_alen,
            "Predicted": arima_preds
        })

        arima_dates = val_len.index if hasattr(val_len, "index") else range(len(val_alen))
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=arima_dates,
            y=val_alen,
            mode='lines',
            name='Actual'
        ))

        fig.add_trace(go.Scatter(
            x=arima_dates,
            y=arima_preds,
            mode='lines',
            name='Predicted'
        ))

        fig.update_layout(
            title="ARIMA Validation: Actual vs Predicted",
            xaxis_title="Date",
            yaxis_title="Cases",
            template="plotly_dark",
            hovermode="x unified"
        )

        st.plotly_chart(fig, width="stretch")

    except Exception as e:
        st.error(f"ARIMA Evaluation Error: {e}")
    cols = st.columns(6)
    with cols[0]:
        st.markdown('<a href="#top">Go to Top</a>', unsafe_allow_html=True)
#Support Vector Regression (SVR)
with tabs[4]:
    st.markdown("## Support Vector Regression (SVR)")

    # Session state sync (same pattern you used above)
    if "svr_days" not in st.session_state:
        st.session_state.svr_days = 30

    def _sync_slider_svr():
        st.session_state.svr_days = st.session_state.svr_slider

    def _sync_input_svr():
        st.session_state.svr_days = st.session_state.svr_input

    st.slider(
        "Select Forecast Days",
        30, 3285,
        value=st.session_state.svr_days,
        key="svr_slider",
        on_change=_sync_slider_svr,
    )

    st.number_input(
        "Enter Forecast Days",
        min_value=30, max_value=3285, step=1,
        value=st.session_state.svr_days,
        key="svr_input",
        on_change=_sync_input_svr,
    )

    svr_days = st.session_state.svr_days

    if st.button("Generate Forecast", key="svr_button"):

        n_lags = len(LAGS)

        preds_scaled = []
        hist = list(svr_history.copy())

        for _ in range(svr_days):
            X = np.array(hist[-n_lags:]).reshape(1, -1)

            yhat = svr.predict(X)[0]

            preds_scaled.append(yhat)
            hist.append(yhat)
        preds_scaled = np.array(preds_scaled).reshape(-1, 1)

            # Inverse scaling
        preds = scaler.inverse_transform(preds_scaled).flatten()
            # Create dates
        svr_last_date = pd.Timestamp(svr_state["last_date"])
        svr_dates = pd.date_range(
                svr_last_date + pd.Timedelta(days=1),
                periods=svr_days,
                freq="D"
            )

        forecast_series = pd.Series(preds, index=svr_dates)
        forecast_series = pd.Series(forecast_series).rolling(7, center=True).mean()
        forecast_series= forecast_series.dropna()
        
        forecast_series = forecast_series.round()
        
            # Plot
        st.line_chart(forecast_series)

            # Table
        forecast_df = forecast_series.reset_index()
        forecast_df.columns = ["Date", "Predicted Cases"]
        forecast_df["Date"] = forecast_df["Date"].dt.date
        forecast_df = forecast_df.astype({"Predicted Cases": int})
        forecast_df = forecast_df.set_index("Date")

        st.dataframe(forecast_df)
        st.download_button(
        "Download Prediction CSV",
        forecast_df.to_csv(index=False).encode("utf-8"),
        "svr_pred.csv",
        "text/csv"
    )
    st.markdown("### SVR Evaluation")
    try:
        train_series = np.asarray(train_len)
        val_series   = np.asarray(val_len)

        svr_preds = compute_svr_eval_cached(
            train_series,
            val_series,
            svr_history,
            scaler,
            svr,
            LAGS
        )
        # Metrics
        svr_rmse = np.sqrt(mean_squared_error(val_series, svr_preds))
        svr_mae  = mean_absolute_error(val_series, svr_preds)


        # Comparison plot
        compare_df = pd.DataFrame({
            "Actual": val_series,
            "Predicted": svr_preds
        })

        svr_dates = val_len.index if hasattr(val_len, "index") else range(len(val_series))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=svr_dates,
            y=val_series,
            mode='lines',
            name='Actual'
        ))

        fig.add_trace(go.Scatter(
            x=svr_dates,
            y=svr_preds,
            mode='lines',
            name='Predicted'
        ))

        fig.update_layout(
            title="SVR Validation: Actual vs Predicted",
            xaxis_title="Date",
            yaxis_title="Cases",
            template="plotly_dark",
            hovermode="x unified"
        )

        st.plotly_chart(fig, width="stretch")
        st.caption("Note: A rolling average is applied for smoother visualization. Evaluation metrics are computed on raw predictions.")
    except Exception as e:
        st.error(f"SVR Evaluation Error: {e}")
    cols = st.columns(6)
    with cols[0]:
        st.markdown('<a href="#top">Go to Top</a>', unsafe_allow_html=True)
#Fourier Wave Model
with tabs[5]:
    st.markdown("## Fourier Wave Model")

    # Session state init
    if "fourier_days" not in st.session_state:
        st.session_state.fourier_days = 30

    # Sync functions
    def _sync_slider_fourier():
        st.session_state.fourier_days = st.session_state.fourier_slider

    def _sync_input_fourier():
        st.session_state.fourier_days = st.session_state.fourier_input

    # Controls
    st.slider(
        "Select Forecast Days",
        30, 3285,
        value=st.session_state.fourier_days,
        key="fourier_slider",
        on_change=_sync_slider_fourier,
    )

    st.number_input(
        "Enter Forecast Days",
        min_value=30, max_value=3285, step=30,
        value=st.session_state.fourier_days,
        key="fourier_input",
        on_change=_sync_input_fourier,
    )

    fourier_days = st.session_state.fourier_days

    t_future = np.arange(
        series_len,
        series_len + fourier_days,
        dtype=float
    )


    if st.button("Run Forecast", key="fourier_button"):

        # Predict
        y_fourier = np.clip(
            wave_model(t_future, *params),
            0, None
        )

        # Date index
        fourier_dates = pd.date_range(
            fourier_last_date + pd.Timedelta(days=1),
            periods=fourier_days,
            freq="D"
        )

        fourier_series = pd.Series(y_fourier, index=fourier_dates)


        fig = go.Figure()

        # Forecast line
        fig.add_trace(go.Scatter(
            x=fourier_series.index,
            y=fourier_series.values,
            mode='lines',
            name='Forecast',
            line=dict(width=3)
        ))

        # Confidence band (if available)
        if fourier_residual_std is not None:
            upper = y_fourier + 1.96 * fourier_residual_std
            lower = y_fourier - 1.96 * fourier_residual_std

            fig.add_trace(go.Scatter(
                x=fourier_dates,
                y=upper,
                line=dict(width=0),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=fourier_dates,
                y=lower,
                fill='tonexty',
                name='Confidence Band',
                line=dict(width=0)
            ))

        fig.update_layout(
            title="COVID-19 Forecast (Fourier Model)",
            xaxis_title="Date",
            yaxis_title="Daily Cases",
            template="plotly_dark",
            hovermode="x unified"
        )

        st.plotly_chart(fig, width="stretch")

    
        df = fourier_series.round().astype(int).reset_index()
        df.columns = ["Date", "Predicted Cases"]
        df["Date"] = df["Date"].dt.date
        df = df.set_index("Date")

        st.dataframe(df)
        st.download_button(
        "Download Prediction CSV",
        df.to_csv(index=False).encode("utf-8"),
        "fourier_pred.csv",
        "text/csv"
    )
    def fourier_predict_range(start_idx, length, params):
        t = np.arange(start_idx, start_idx + length, dtype=float)
        return np.clip(wave_model(t, *params), 0, None)
    st.markdown("### Fourier Model Validation and Evaluation")

    try:

        fourier_preds = fourier_eval_cached(len(train_len), len(val_len), params)
        fourier_rmse = np.sqrt(mean_squared_error(val_len, fourier_preds))
        fourier_mae  = mean_absolute_error(val_len, fourier_preds)

        compare_df = pd.DataFrame({
            "Actual": val_len,
            "Predicted": fourier_preds
        })

        fourier_dates = val_len.index if hasattr(val_len, "index") else range(len(val_len))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fourier_dates,
            y=val_len,
            mode='lines',
            name='Actual'
        ))

        fig.add_trace(go.Scatter(
            x=fourier_dates,
            y=fourier_preds,
            mode='lines',
            name='Predicted'
        ))

        fig.update_layout(
            title="Fourier Model Validation: Actual vs Predicted",
            xaxis_title="Date",
            yaxis_title="Cases",
            template="plotly_dark",
            hovermode="x unified"
        )

        st.plotly_chart(fig, width="stretch")

    except Exception as e:
        st.error(f"Fourier Evaluation Error: {e}")
    cols = st.columns(6)
    with cols[0]:
        st.markdown('<a href="#top">Go to Top</a>', unsafe_allow_html=True)
with tabs[6]:
    st.markdown("## Models Comparison")

    try:
        models_val = {}

        if 'hw_preds' in locals():
            models_val['Holt-Winters'] = (hw_preds, hw_rmse, hw_mae)

        if 'arima_preds' in locals():
            models_val['ARIMA(7,1,3)'] = (arima_preds, arima_rmse, arima_mae)

        if 'svr_preds' in locals():
            models_val['SVR (RBF)'] = (svr_preds, svr_rmse, svr_mae)

        if 'fourier_preds' in locals():
            models_val['Fourier (vanilla)'] = (fourier_preds, fourier_rmse, fourier_mae)
    
        suppression = (
            suppression_forecast
            .reindex(val_len.index)
            .ffill()
            .fillna(0)
            .values
        )

        y_vacc_adj_val = fourier_preds * (1 - suppression * VACC_SENSITIVITY)
        y_vacc_adj_val = np.maximum(y_vacc_adj_val, 0)

        fv_rmse = np.sqrt(mean_squared_error(val_len.values, y_vacc_adj_val))
        fv_mae  = mean_absolute_error(val_len.values, y_vacc_adj_val)

        models_val['Fourier + Vaccination Adjusted'] = (y_vacc_adj_val, fv_rmse, fv_mae)

    # Create results DataFrame
        df_results = pd.DataFrame([
            {"Model": name, "RMSE": rmse, "MAE": mae}
            for name, (_, rmse, mae) in models_val.items()
        ])
    # Rank models by RMSE
        df_results = df_results.sort_values("RMSE").reset_index(drop=True)
        df_results["Rank"] = df_results.index + 1

        best_model = df_results.loc[0, "Model"]

    # Highlight best model in the table
        st.dataframe(
            df_results.style.apply(
                lambda row: ["background-color: #1e3a8a; color: white; font-weight: bold;" if row.name == 0 else "" for _ in row],
                axis=1
            ),
            use_container_width=True
        )

        st.success(f" Best short-term model: {best_model}")
        st.info(" Best long-term model: Fourier + Vaccination Adjusted")

        fig_rmse = px.bar(
            df_results,
            x="Model",
            y="RMSE",
            text=df_results["RMSE"].apply(lambda x: f"{x/1e3:.1f}K"),
            title="Validation RMSE"
        )

        fig_mae = px.bar(
            df_results,
            x="Model",
            y="MAE",
            text=df_results["MAE"].apply(lambda x: f"{x/1e3:.1f}K"),
            title="Validation MAE"
        )

        fig_rmse.update_layout(xaxis_tickangle=25)
        fig_mae.update_layout(xaxis_tickangle=25)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig_rmse, width="stretch")

        with col2:
            st.plotly_chart(fig_mae, width="stretch")
        
    except Exception as e:
        st.error(f"Model comparison error: {e}")
    st.markdown(
    "## Forecast Performance Analysis"
    )

    actual_mean = np.mean(val_len.values)

    performance_rows = []

    for name, (_, rmse, mae) in models_val.items():

        performance_score = max(
            0,
            100 - (
                (mae / actual_mean) * 100
            )
        )

        performance_rows.append({

            "Model": name,

            "RMSE": round(rmse, 2),

            "MAE": round(mae, 2),

            "Performance Score (%)": round(
                performance_score,
                2
            )

        })

    performance_df = pd.DataFrame(
        performance_rows
    )

    performance_df = (
        performance_df
        .sort_values(
            "Performance Score (%)",
            ascending=False
        )
        .reset_index(drop=True)
    )

    performance_df["Rank"] = (
        performance_df.index + 1
    )

    best_perf_model = (
        performance_df.iloc[0]["Model"]
    )

    best_perf_score = (
        performance_df.iloc[0]
        ["Performance Score (%)"]
    )

    st.success(
        f"Best Performance Model: "
        f"{best_perf_model} "
        f"({best_perf_score:.2f}%)"
    )

    st.dataframe(

        performance_df.style.apply(

            lambda row: [

                (
                    "background-color: #14532d;"
                    "color: white;"
                    "font-weight: bold;"
                )

                if row.name == 0
                else ""

                for _ in row

            ],

            axis=1

        ),

        use_container_width=True

    )

    fig_perf = px.bar(

        performance_df,

        x="Model",

        y="Performance Score (%)",

        text="Performance Score (%)",

        title="Forecast Performance Score"

    )

    fig_perf.update_traces(

        texttemplate='%{text:.2f}%',

        textposition='outside'

    )

    fig_perf.update_layout(

        template="plotly_dark",

        yaxis_title="Performance Score (%)",

        xaxis_title="Models",

        xaxis_tickangle=20

    )

    st.plotly_chart(
        fig_perf,
        width="stretch"
    )

    st.markdown(
        "### Individual Model Performance"
    )

    cols = st.columns(
        len(performance_df)
    )

    for i, row in performance_df.iterrows():

        with cols[i]:

            st.metric(

                label=row["Model"],

                value=(
                    f"{row['Performance Score (%)']:.2f}%"
                )

            )
    
    st.info("""
    Performance Score is a normalized metric derived from MAE relative to average actual values.

    It provides an intuitive comparison across models,
    but RMSE and MAE remain the primary evaluation metrics.
    """)
    st.info("""
    Fourier + Vaccination Adjusted model is used for long-term forecasting 
    because it captures wave patterns and incorporates real-world vaccination effects.
    """)
    st.markdown("### Final Insight")

    st.success("""
    Model performance depends on how well the model aligns with underlying data patterns.

    In this case, wave-like COVID behavior makes Fourier-based models more suitable
    for long-term forecasting.
    """)
    st.markdown("### Model Parameters Summary")

    param_data = [
        {"Model": "Holt-Winters", "Param": "alpha", "Value": hw_state["alpha"]},
        {"Model": "Holt-Winters", "Param": "beta", "Value": hw_state["beta"]},
        {"Model": "Holt-Winters", "Param": "gamma", "Value": hw_state["gamma"]},

        {"Model": "ARIMA", "Param": "p", "Value": p},
        {"Model": "ARIMA", "Param": "q", "Value": q},
        {"Model": "ARIMA", "Param": "alpha", "Value": alpha},

        {"Model": "SVR", "Param": "lags", "Value": len(LAGS)},
        {"Model": "SVR", "Param": "min", "Value": scaler.data_min_[0]},
        {"Model": "SVR", "Param": "max", "Value": scaler.data_max_[0]},

        {"Model": "Fourier", "Param": "A1", "Value": fourier_state["A1"]},
        {"Model": "Fourier", "Param": "A2", "Value": fourier_state["A2"]},
        {"Model": "Fourier", "Param": "decay", "Value": fourier_state["decay"]},

        {"Model": "Vaccination", "Param": "lag_days", "Value": LAG_DAYS},
        {"Model": "Vaccination", "Param": "sensitivity", "Value": VACC_SENSITIVITY},
    ]

    df_params = pd.DataFrame(param_data)

    st.dataframe(df_params, use_container_width=True)

    # CSV Export
    st.download_button(
        "Download Parameters CSV",
        df_params.to_csv(index=False).encode("utf-8"),
        "model_parameters.csv",
        "text/csv"
    )
    cols = st.columns(6)
    with cols[0]:
        st.markdown('<a href="#top">Go to Top</a>', unsafe_allow_html=True)

#Wave Summary
with tabs[7]:
    st.markdown("## Complete Wave Summary (2020 to 2030)")

    rows = []

    # Historical Waves
    for i, (_, d, v) in enumerate(hist_peaks_info):

        if d < pd.Timestamp("2021-03-08"):
            vacc_info = "Pre-vaccination"
        else:
            try:
                vacc_val = merged.loc[d, "pct_dose1"]
                vacc_info = f"{vacc_val:.1f}%"
            except:
                vacc_info = "N/A"

        rows.append({
            "Wave": f"Wave {i+1}",
            "Status": "Confirmed",
            "Peak Date": d.strftime("%d %B %Y"),
            "Peak Cases/Day": int(v),
            "Vacc Coverage (Dose-1)": vacc_info,
            "Source": "Historical Data"
        })


    # Future Waves 
    w_num = len(hist_peaks_info) + 1

    for i, (_, d, v) in enumerate(future_peaks_vacc[:10]):

        # Future vaccination = last known value (safe assumption)
        try:
            last_vacc = merged["pct_dose1"].iloc[-1]
            vacc_info = f"{last_vacc:.1f}% (latest)"
        except:
            vacc_info = "Not available"

        rows.append({
            "Wave": f"Wave {w_num+i}",
            "Status": "Predicted",
            "Peak Date": d.strftime("%d %B %Y"),
            "Peak Cases/Day": int(v),
            "Source": "Fourier + Vaccination Adjusted"
        })


    # Create DataFrame
    wave_df = pd.DataFrame(rows)

    # Display with conditional formatting
    st.dataframe(
        wave_df.style.apply(
            lambda row: [
                "background-color: #14532d; color: white;"  # dark green
                if "Confirmed" in row["Status"]
                else "background-color: #78350f; color: white;"  # dark amber
                for _ in row
            ],
            axis=1
        ),
        use_container_width=True
    )

    st.download_button(
        "Download Wave Predictions CSV",
        wave_df.to_csv(index=False).encode("utf-8"),
        "wave_predictions_2030.csv",
        "text/csv"
    )
    prevented = (fourier_vanilla - fourier_vacc_adj).sum()

    st.metric("Estimated Cases Prevented", f"{int(prevented):,}")
    st.info("These are the estimated cases that would have occurred without vaccination drive.")
    st.markdown("## Final Prediction Insights")

    if future_peaks_vacc:
        next_date = future_peaks_vacc[0][1]
        next_cases = future_peaks_vacc[0][2]
        interval = (next_date - hist_peaks_info[-1][1]).days

        next_date_str = next_date.strftime('%B %Y')
    else:
        next_date_str = "N/A"
        next_cases = 0
        interval = 0

    st.success(f"""
    - Next wave is estimated around: {next_date_str}

    - Expected peak cases: {int(next_cases):,} per day

    - Inter-wave interval: {interval} days

    - Vaccination significantly reduces peak intensity

    Long-term trend shows gradual decline due to immunity + vaccination
    """)
    st.info("Forecasts represent possible scenarios based on historical patterns, not exact future outcomes.")
    cols = st.columns(6)
    with cols[0]:
        st.markdown('<a href="#top">Go to Top</a>', unsafe_allow_html=True)
cols = st.columns(6)
with cols[0]:
    if st.button("Back"):
        st.switch_page("pages/5_Models.py")
with cols[5]:
    if st.button("Next"):
        st.switch_page("pages/About.py")
st.markdown("---")
