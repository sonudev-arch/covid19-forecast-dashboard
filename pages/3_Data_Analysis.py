import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.signal import find_peaks
from plotly.subplots import make_subplots

from utils.data_loader import (
    load_covid_data,
    load_vaccination_data,
    load_vaccination2_data,
    load_merged
)
st.set_page_config(
    page_title="Data Analysis",
    layout="wide",
    
)
@st.cache_data
def load_data():
#Correcting Spelling and other Mistakes
    data = load_covid_data()
    data["State/UnionTerritory"]=data["State/UnionTerritory"].replace(['Bihar****'],'Bihar')
    data["State/UnionTerritory"]=data["State/UnionTerritory"].replace(['Maharashtra***'],'Maharashtra')
    data["State/UnionTerritory"]=data["State/UnionTerritory"].replace(['Madhya Pradesh***'],'Madhya Pradesh')
    data["State/UnionTerritory"]=data["State/UnionTerritory"].replace(['Dadra and Nagar Haveli'],'Dadra and Nagar Haveli and Daman and Diu')
    data["State/UnionTerritory"]=data["State/UnionTerritory"].replace(['Daman & Diu'],'Dadra and Nagar Haveli and Daman and Diu')
    data["State/UnionTerritory"]=data["State/UnionTerritory"].replace(['Himanchal Pradesh'],'Himachal Pradesh')
    data["State/UnionTerritory"]=data["State/UnionTerritory"].replace(['Karanataka'],'Karnataka')
    data["State/UnionTerritory"]=data["State/UnionTerritory"].replace(['Telengana'],'Telangana')
#Dropping Unwanted rows
    data = data[~data['State/UnionTerritory'].isin(['Cases being reassigned to states','Unassigned'])]
    return data
data=load_data()

@st.cache_data
def load_merged_data():
    merged = load_merged()
    merged['Date'] = pd.to_datetime(merged['Date'])
    merged = merged.set_index('Date')
    merged['NewDeaths_7d'] = merged['NewDeaths'].rolling(7, min_periods=1).mean()
    merged['CFR_smooth'] = merged['CFR_pct'].rolling(14, min_periods=1).mean()
    return merged
merged = load_merged_data()

tabs = st.tabs([
    "Overview",
    "Trends",
    "Top States",
    "Active Cases",
    "Growth Rate",
    "Correlation",
    "Vaccination",
    "EDA"
])
with tabs[0]:

    st.title("Data Analysis")
    st.markdown("---")
    st.markdown("## Overview")
    st.write("""
    Data analysis is the systematic process of collecting, cleaning, transforming, and interpreting data
    to extract meaningful insights, identify patterns, and support decision-making.

    In this project, data analysis is performed on COVID-19 time-series data to understand how the pandemic evolved,
    analyze trends across states, and study the impact of various factors such as vaccination.
    """)

    st.markdown("---")

    st.markdown("## Data Analysis Overview")

    st.write("""
    This section provides a comprehensive analytical view of COVID-19 trends across India.
    It combines case data, recovery patterns, mortality trends, and vaccination progress
    to build a unified understanding of the pandemic.
    """)

    st.markdown("---")

    st.markdown("### Exploratory Data Analysis")

    st.write("""
    Exploratory Data Analysis (EDA) is the process of examining and visualizing data to understand its structure,
    identify patterns, detect anomalies, and generate initial insights.

    In this project, EDA is used to:
    - Explore time-series behavior of COVID-19 cases  
    - Identify peaks and wave patterns  
    - Detect irregularities and sudden changes in reporting  
    - Compare trends across states and regions  
    - Understand variability and distribution in the data  

    EDA forms the foundation for further analysis and forecasting.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### What This Section Covers")

        st.markdown("""
        - Epidemic trends showing progression of confirmed, recovered, and death cases  
        - State-wise analysis for regional comparison  
        - Active case monitoring to assess healthcare burden  
        - Growth rate analysis using percentage change and smoothing techniques  
        - Vaccination analysis to study its impact on case trends  
        """)

    with col2:
        st.markdown("### Key Insights")

        st.markdown("""
        - Identification of infection peaks and wave patterns  
        - Comparison of performance across states  
        - Observation of changes in fatality trends  
        - Analysis of the relationship between vaccination and cases  
        - Detection of volatility and irregular behavior in data  
        """)

    st.markdown("---")

    st.markdown("### Why This Analysis Matters")

    col3, col4, col5 = st.columns(3)

    col3.info("Supports data-driven decision making")
    col4.info("Helps identify patterns and anomalies")
    col5.info("Provides clear and interpretable insights")

    st.markdown("---")

    st.markdown("### How to Use")

    st.markdown("""
    - Use filters to select states, metrics, and time ranges  
    - Navigate tabs to explore different analytical views  
    - Hover on charts for detailed values  
    - Compare metrics to understand relationships  
    """)

    st.markdown("---")

    st.warning("""
    Insights are based on historical data and statistical smoothing techniques.
    They help understand past trends but should be interpreted cautiously for future predictions.
    """)
    
with tabs[1]:
    st.markdown("# Trend Analysis")
 
    st.write('Trend analysis is a statistical approach to identifying patterns or changes in data over time. It\'s used to help predict future business dynamics and inform decision-making. Whether used for finance, marketing, supply chain management, economics, healthcare or environmental sciences, it can be a useful tool for any organization looking to build evidence-based strategies based on historical precedents.')
    st.markdown('## Trend Analysis overview')
    st.markdown("""
    Trend analysis examines how COVID-19 cases evolved over time, helping identify
    patterns such as growth phases, peaks, and decline periods. This enables a deeper
    understanding of how the pandemic progressed across regions.
    """)
    
    data["Date"] = pd.to_datetime(data["Date"])
    # Important controls FIRST
    metrics = st.multiselect(
        "Select Metrics for Daily Trends",
        ["Confirmed", "Cured", "Deaths"],
        default=["Confirmed"]
    )
    # Compare states
    states = ["India"] + sorted(data["State/UnionTerritory"].unique())
    selected_state = st.selectbox(
        " Select State",
        states
    )

    if selected_state == "India":
        # Aggregate whole country
        filtered_data = data.groupby("Date")[["Confirmed", "Cured", "Deaths"]].sum().reset_index()
    else:
        # Filter specific state
        filtered_data = data[data["State/UnionTerritory"] == selected_state]

    # Available dates
    available_dates = sorted(data["Date"].unique())

    
    start_date=available_dates[0]
    end_date =  available_dates[-1]
   
    st.markdown("####")

    if selected_state == "India":
        temp = data.copy()
    else:
        temp = data[data["State/UnionTerritory"] == selected_state]

    filtered_data = temp[
        (temp["Date"] >= start_date) &  
        (temp["Date"] <= end_date)
    ]

    st.subheader(f"Data for {selected_state}")

    # Aggregate trend
    trend = filtered_data.groupby("Date")[metrics].sum().reset_index()

    # Create figure
    fig = go.Figure()

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=trend["Date"],
            y=trend[metric],
            mode='lines',
            name=metric,
            line=dict(color=colors[i], width=3)
        ))

    # Layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend_title="Metrics"
    )

    # Add range slider and buttons
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    # Show chart
    st.plotly_chart(fig, use_container_width=True)

    # Caption
    st.caption(
        f"**:blue[Showing trends for {', '.join(metrics)} in {selected_state} "
        f"from {start_date.date()} to {end_date.date()}]**"
    )
    compare_mode = st.checkbox("Compare 2 States")
    
    state1 = st.selectbox(
            "Select First State",
            states,
        )
    if compare_mode:
            state2 = st.selectbox(
            "Select Second State",
            states,
        )
    show_daily = st.checkbox("Show Daily Cases", True)
    show_ma = st.checkbox("Show 7-Day Moving Avg", True)
    show_peaks = st.checkbox("Show Peaks", True)
    
    def prepare_trend(df, state):
        temp = df if state == "India" else df[df["State/UnionTerritory"] == state]
        
        trend = temp.groupby("Date")[metrics].sum().reset_index()
        
        # Daily cases
        for m in metrics:
            trend[f"{m}_daily"] = trend[m].diff().fillna(0)
        
        # Moving average
        for m in metrics:
            trend[f"{m}_ma"] = trend[f"{m}_daily"].rolling(7).mean()
        
        return trend

    def get_peaks(series):
        peaks, _ = find_peaks(series, distance=7, prominence=series.mean()*0.3)
        return peaks

    fig = go.Figure()

    # Primary state
    trend = prepare_trend(data, selected_state)
       # Comparison state
    
    trend1 = prepare_trend(data,state1)

    for i, metric in enumerate(metrics):
        
        # Daily
        if show_daily:
            fig.add_trace(go.Scatter(
                x=trend1["Date"],
                y=trend1[f"{metric}_daily"],
                mode='lines',
                name=f"{state1} {metric} (Daily)",
                
            ))
        
        # Moving Average
        if show_ma:
            fig.add_trace(go.Scatter(
                x=trend1["Date"],
                y=trend1[f"{metric}_ma"],
                mode='lines',
                name=f"{state1} {metric} (7-day MA)",
                line=dict(width=3, dash='dash')
            ))
        
        # Peaks
        if show_peaks:
            peak_idx = get_peaks(trend1[f"{metric}_daily"])
            
            fig.add_trace(go.Scatter(
                x=trend1["Date"].iloc[peak_idx],
                y=trend1[f"{metric}_daily"].iloc[peak_idx],
                mode='markers',
                name=f"{state1} Peaks",
                marker=dict(color='red', size=8)
            ))
    
    
    if compare_mode:
        trend2 = prepare_trend(data, state2)
      
        for i, metric in enumerate(metrics):
            
            # Daily
            if show_daily:
                fig.add_trace(go.Scatter(
                    x=trend2["Date"],
                    y=trend2[f"{metric}_daily"],
                    mode='lines',
                    name=f"{state2} {metric} (Daily)",
                    
                ))
            
            # Moving Average
            if show_ma:
                fig.add_trace(go.Scatter(
                    x=trend2["Date"],
                    y=trend2[f"{metric}_ma"],
                    mode='lines',
                    name=f"{state2} {metric} (7-day MA)",
                    line=dict(width=3, dash='dash')
                ))
            
            # Peaks
            if show_peaks:
                peak_idx = get_peaks(trend2[f"{metric}_daily"])
                
                fig.add_trace(go.Scatter(
                    x=trend2["Date"].iloc[peak_idx],
                    y=trend2[f"{metric}_daily"].iloc[peak_idx],
                    mode='markers',
                    name=f"{state2} Peaks",
                    marker=dict(color='red', size=8)
                ))
        



    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## Key Insights")

    col1, col2 = st.columns(2)

    # LEFT SIDE INSIGHTS
    with col1:
        st.info("""
        **Growth Phase:**  
        Initial periods show a steady rise in cases, indicating rapid transmission
        and limited containment measures.
        """)

        st.info("""
        **Peak Detection:**  
        Sharp spikes represent pandemic waves where infection rates were highest.
        These peaks help identify critical stress periods for healthcare systems.
        """)

        st.info("""
        **Decline Phase:**  
        After peaks, a downward trend indicates recovery due to interventions such
        as lockdowns, awareness, and vaccination efforts.
        """)

    # RIGHT SIDE INSIGHTS
    with col2:
        st.info("""
        **Moving Average Stability:**  
        The 7-day moving average smooths fluctuations and reveals the true trend,
        eliminating daily reporting noise.
        """)

        st.info("""
        **Volatility Observation:**  
        Large fluctuations in daily cases indicate unstable spread dynamics and
        possible external influences like policy changes or new variants.
        """)

        st.info(f"""
        **State-Level Behavior:**  
        Trends for **{selected_state}** may differ significantly from national patterns,
        highlighting regional differences in spread and control measures.
        """)

    st.markdown("---")

    st.markdown("### Interpretation Summary")

    st.success(f"""
    - The trend shows how **{selected_state}** experienced multiple phases of the pandemic.  
    - Peaks indicate **wave intensity**, while declines reflect **effective interventions**.  
    - Moving averages confirm the **overall direction of spread** beyond daily noise.  
    - Understanding these patterns helps in **predicting future risks and preparing responses**.
    """)

    st.markdown("---")

    # Optional highlight box
    st.warning("""
    **Note:** Trend analysis is based on historical data. Sudden changes in policy,
    behavior, or virus variants can alter future patterns significantly.
    """)
    @st.cache_data
    def get_top_10(data, metric):
        df = (
            data.groupby("State/UnionTerritory")[metric]
            .max()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        df.columns = ["State", metric]

        df["Rank"] = (
            df[metric]
            .rank(method='dense', ascending=False)
            .astype(int)
        )

        return df    
#Top 10 States/UT Analysis
with tabs[2]:

    st.markdown("## Top 10 States / UT")

    st.write("""
    This section highlights the most affected states based on selected metrics,
    helping identify regional impact and severity of the pandemic.
    """)


    col1, col2 = st.columns([2, 1])

    with col2:
        metric = st.selectbox(
            "Select Metric",
            ["Confirmed", "Cured", "Deaths"]
        )

 


    top_10_df = get_top_10(data, metric)

   
    with col1:
        st.subheader(f"Top 10 States by {metric} cases")

        fig = px.bar(
            top_10_df.sort_values(metric),
            x=metric,
            y="State",
            orientation='h',
            text=metric,
            color=metric,
            color_continuous_scale="teal"
        )

        fig.update_layout(
            height=450,
            template="plotly_white",
            margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_showscale=False
        )

        fig.update_traces(
            texttemplate='%{text:,}',
            textposition='outside'
        )

        st.plotly_chart(fig, use_container_width=True)

 
    col3, col4 = st.columns([2, 1])

    with col3:
        table_df = top_10_df.copy()
        table_df[metric] = table_df[metric].apply(lambda x: f"{x:,}")
        table_df = table_df.sort_values("Rank").set_index("Rank")

        st.dataframe(table_df, use_container_width=True)

    with col4:
        worst_state = top_10_df.iloc[0]["State"]

        st.metric(
            " Worst Affected State",
            worst_state
        )

        st.info(f"{worst_state} shows the highest {metric.lower()} count.")

    st.markdown("---")

@st.cache_data
def load_active():
        data = load_data()
        data = data[["Date","Confirmed","Cured","Deaths"]]

        data["Date"] = pd.to_datetime(data["Date"])
        data = data.groupby("Date").sum()

    # Calculate Active
        data["Active"] = data["Confirmed"] - data["Cured"] - data["Deaths"]

    # Reset index for Plotly
        data = data.reset_index()
        return data
#Active Cases Analysis
with tabs[3]:

    st.markdown("## Active Cases Analysis")

    st.write("""
    Active cases represent the number of ongoing infections at any given time.
    This metric reflects the real-time burden on the healthcare system and helps
    identify peak stress periods during the pandemic.
    """)

    df = load_active()
    peaks, _ = find_peaks(
    df["Active"],
    distance=60,                      # avoid close peaks
    prominence=df["Active"].mean()*0.3
)

# Get peak values
    peak_df = df.iloc[peaks].sort_values("Active", ascending=False)

    if len(peak_df) >= 2:
        wave2 = peak_df.iloc[0]   # highest peak
        wave1 = peak_df.iloc[1]   # second highest

        wave2_peak = wave2["Active"]
        wave2_date = wave2["Date"]

        wave1_peak = wave1["Active"]
        wave1_date = wave1["Date"]
        
        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Wave 1 Peak Cases", f"{wave1_peak:,.0f}")
        k2.metric("Wave 1 Peak Date", wave1_date.strftime("%d %b %Y"))

        k3.metric("Wave 2 Peak Cases", f"{wave2_peak:,.0f}")
        k4.metric("Wave 2 Peak Date", wave2_date.strftime("%d %b %Y"))
        st.markdown("---")
    else:
        st.warning("Not enough peaks detected.")


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Active"],
        mode='lines',
        name='Active Cases',
        line=dict(color="#ff7715", width=3),
        fill='tozeroy',
        fillcolor='rgba(255,119,21,0.15)'
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Active Cases",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        margin=dict(l=10, r=10, t=40, b=10)
    )
        # Wave 1 marker
    fig.add_trace(go.Scatter(
        x=[wave1_date],
        y=[wave1_peak],
        mode='markers+text',
        text=["Wave 1"],
        textposition="top center",
        marker=dict(color='blue', size=10),
        name="Wave 1 Peak"
    ))

    # Wave 2 marker
    fig.add_trace(go.Scatter(
        x=[wave2_date],
        y=[wave2_peak],
        mode='markers+text',
        text=["Wave 2"],
        textposition="top center",
        marker=dict(color='red', size=10),
        name="Wave 2 Peak"
    ))
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
**Wave Comparison:**  
- Wave 1 peaked at **{wave1_peak:,.0f}** cases on **{wave1_date.strftime("%d %B %Y")}**  
- Wave 2 peaked at **{wave2_peak:,.0f}** cases on **{wave2_date.strftime("%d %B %Y")}**

Wave 2 was **{wave2_peak / wave1_peak:.1f}x more severe**, indicating a significantly higher transmission rate.
""")

        st.info("""
        **Trend Behavior:**  
        A rising curve indicates increasing spread, while a decline reflects
        recovery and effective containment measures.
        """)

    with col2:
        st.info("""
        **Interpretation:**  
        Sustained decline suggests stabilization, while sudden spikes may indicate
        new outbreaks or reporting anomalies.
        """)

    st.markdown("---")

    st.warning("""
    Active cases depend on reporting accuracy and recovery definitions.
    Sudden changes may reflect data updates rather than real-world shifts.
    """)

#Growth Rate Analysis
@st.cache_data
def load_growth(data):
    df = data.groupby('Date')[['Confirmed', 'Cured', 'Deaths']].sum().reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # Growth %
    df['Confirmed Growth'] = df['Confirmed'].pct_change() * 100
    df['Cured Growth'] = df['Cured'].pct_change() * 100
    df['Deaths Growth'] = df['Deaths'].pct_change() * 100

    # Moving averages
    for col in ["Confirmed Growth", "Cured Growth", "Deaths Growth"]:
        df[f"{col} MA"] = df[col].rolling(7).mean()
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df.fillna(0)

with tabs[4]:

    st.markdown("## Growth Rate Analysis")

    st.write("""
    Growth rate reflects how quickly COVID-19 cases, recoveries, or deaths are changing.
    It helps identify acceleration, stabilization, or decline phases of the pandemic.
    """)

    data_growth = load_growth(data)
    metric = st.selectbox(
        "Select Metric",
        ["Confirmed Growth", "Cured Growth", "Deaths Growth"]
    )

    avg_growth = data_growth[metric].mean()
    volatility = data_growth[metric].std()
    current_growth = data_growth[metric].iloc[-1]

    k1, k2, k3 = st.columns(3)

    k3.metric("Latest Growth Rate (Dataset)", f"{current_growth:.2f}%")
    k1.metric("Avg Growth", f"{avg_growth:.2f}%")
    k2.metric("Volatility", f"{volatility:.2f}")
    st.markdown("---")


    fig = go.Figure()

        # Raw
    fig.add_trace(go.Scatter(
            x=data_growth.index,
            y=data_growth[metric],
            mode='lines',
            name=f"{metric} (Raw)",
            line=dict(width=2)
        ))

        # MA
    fig.add_trace(go.Scatter(
            x=data_growth.index,
            y=data_growth[f"{metric} MA"],
            mode='lines',
            name=f"{metric} (7-day MA)",
            line=dict(width=3, dash='dash')
        ))

    # Layout (only once)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Growth Rate (%)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    st.plotly_chart(fig, use_container_width=True)


    st.markdown("### Insights")

    col1, col2 = st.columns(2)

    with col1:

        st.info("""
        **Moving Average Insight:**  
        The 7-day moving average smooths fluctuations and shows the true trend,
        removing daily reporting noise.
        """)
        st.info(f"""
        **Volatility Analysis:**  
        Volatility of **{volatility:.2f}** indicates how unstable the spread is.
        Higher values mean unpredictable spikes.
        """)
    with col2:
        

        st.info("""
        **Interpretation:**  
        - Positive growth → expansion phase  
        - Negative growth → contraction phase  
        - Near zero → stabilization
        """)

    st.markdown("---")

    st.warning("""
    Growth rates can be sensitive to sudden reporting changes or data corrections.
    Interpret short-term spikes cautiously.""")
#Correlation Analysis
with tabs[5]:

    st.markdown("## Correlation Analysis")

    st.write("""
    Correlation measures the strength and direction of relationships between variables.
    Values range from **-1 to +1**:
    - **+1** → strong positive relationship  
    - **0** → no relationship  
    - **-1** → strong negative relationship  
    """)


    corr = data[["Confirmed", "Cured", "Deaths"]].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### Insights")

    # Extract key correlations
    conf_cured = corr.loc["Confirmed", "Cured"]
    conf_deaths = corr.loc["Confirmed", "Deaths"]
    cured_deaths = corr.loc["Cured", "Deaths"]

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Confirmed vs Cured:**  
        Correlation = **{conf_cured:.2f}**  
        Indicates that as confirmed cases increase, recoveries also increase.
        """)

        st.info(f"""
        **Confirmed vs Deaths:**  
        Correlation = **{conf_deaths:.2f}**  
        Suggests fatalities rise with increasing infections.
        """)

    with col2:
        st.info(f"""
        **Cured vs Deaths:**  
        Correlation = **{cured_deaths:.2f}**  
        Shows both outcomes scale with overall case volume.
        """)

        st.info("""
        **Interpretation:**  
        High positive correlations are expected since all variables depend on total infections.
        """)

    st.markdown("---")

    st.warning("""
    Correlation does not imply causation.  
    These relationships exist because all variables are derived from total cases,
    not because one directly causes another.
    """)

with tabs[6]:

    st.markdown("## Vaccination Drive")

    st.write("""
    This section analyzes the distribution of COVID-19 vaccines across Indian states
    and the contribution of different vaccine types in the overall rollout.
    """)

    vacc_data = load_vaccination_data()

    vacc_data = vacc_data.rename(columns={
        'Total Achievement - Total Doses Administered': 'Total Doses'
    })

    doses = vacc_data[['State/UT', 'Total Doses']]
    doses = doses[~doses['State/UT'].isin(['Total'])]

   
    total_doses = doses["Total Doses"].sum()
    top_state = doses.sort_values("Total Doses", ascending=False).iloc[0]

    k1, k2 = st.columns(2)

    k1.metric("Total Doses Administered", f"{total_doses:,.0f}")
    k2.metric("Top Vaccinated State", top_state["State/UT"])

    st.markdown("---")

    option = st.radio("Mode", ["Top 10", "All"], horizontal=True)

    if option == "Top 10":
        display_data = doses.sort_values("Total Doses").tail(10)
    else:
        display_data = doses.sort_values("Total Doses")


    fig_bar = px.bar(
        display_data,
        x='Total Doses',
        y="State/UT",
        orientation='h',
        text='Total Doses',
        color='Total Doses',
        color_continuous_scale="teal",
        height=600
    )

    fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')

    fig_bar.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")


    @st.cache_data
    def load_vaccine_types():
        df = load_vaccination2_data()
        df['covishield'] = pd.to_numeric(df['covishield'], errors='coerce')
        df['covaxin'] = pd.to_numeric(df['covaxin'], errors='coerce')
        df['sputnik'] = pd.to_numeric(df['sputnik'], errors='coerce')
        return df

    vacc_type = load_vaccine_types()

    covishield_total = vacc_type['covishield'].sum()
    covaxin_total = vacc_type['covaxin'].sum()
    sputnik_total = vacc_type['sputnik'].sum()


    labels = ["Covishield", "Covaxin", "Sputnik"]
    values = [covishield_total, covaxin_total, sputnik_total]

    fig_pie = px.pie(
        names=labels,
        values=values,
        title="Vaccine Type Distribution",
        color=labels,
        color_discrete_map={
            "Covishield": "#1f77b4",
            "Covaxin": "#2ca02c",
            "Sputnik": "#d62728"
        }
    )

    fig_pie.update_traces(
        textinfo='percent+label',
        pull=[0.05, 0, 0],
        hole=0.3
    )

    st.plotly_chart(fig_pie, use_container_width=True)


    st.markdown("### Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Vaccination Coverage:**  
        A total of **{total_doses:,.0f} doses** have been administered across India,
        indicating large-scale immunization efforts.
        """)

        st.info(f"""
        **Top Performing State:**  
        **{top_state['State/UT']}** leads in vaccination rollout,
        reflecting strong distribution infrastructure.
        """)

    with col2:
        st.info("""
        **Vaccine Dominance:**  
        Covishield contributes the largest share, dominating the vaccination drive.
        """)

        st.info("""
        **Interpretation:**  
        Uneven distribution across states may indicate differences in population,
        logistics, or policy effectiveness.
        """)

    st.markdown("---")

with tabs[7]:
#EDA of merged dataset
    st.markdown("## Exploratory Data Analysis")
    st.write('Exploratory Data Analysis (EDA) is the process of analyzing and visualizing data to uncover patterns, relationships, and insights. It helps to understand the structure of the data, identify anomalies, and formulate hypotheses for further analysis.')
    st.markdown("### Merged COVID-19 & Vaccination Data")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "A — New Confirmed Cases",
            "B — New Deaths (7d MA)",
            "C — Active Cases",
            "D — Case Fatality Rate (%)"
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    fig.add_trace(
        go.Bar(
            x=merged.index,
            y=merged['NewConfirmed'],
            name="Raw daily",
            marker_color='rgba(255,0,0,0.3)'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged['NewConfirmed_7d'],
            name="7d MA",
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged['NewConfirmed_14d'],
            name="14d MA",
            line=dict(color='darkblue', dash='dash')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged['NewDeaths'],
            fill='tozeroy',
            name="Daily deaths",
            line=dict(color='purple')
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged['NewDeaths_7d'],
            name="7d MA",
            line=dict(color='purple', width=2)
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged['ActiveCases'],
            fill='tozeroy',
            name="Active cases",
            line=dict(color='orange')
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged['CFR_smooth'],
            name="CFR (smoothed)",
            line=dict(color='navy')
        ),
        row=2, col=2
    )

    # Mean line
    mean_cfr = merged['CFR_pct'].mean()

    fig.add_hline(
        y=mean_cfr,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean = {mean_cfr:.2f}%",
        row=2, col=2
    )
    fig.update_layout(
        title="India COVID-19 — National Daily Overview (Jan 2020 - Aug 2021)",
        height=700,
        template="plotly_white",
        showlegend=True
    )
    peak_idx = merged['NewConfirmed_7d'].idxmax()
    peak_val = merged['NewConfirmed_7d'].max()

    fig.add_annotation(
        x=peak_idx,
        y=peak_val,
        text=f"Peak: {int(peak_val):,}",
        showarrow=True,
        arrowhead=2,
        row=1, col=1
    )
    # Cases -> Lakhs
    fig.update_yaxes(tickformat=".2s", row=1, col=1)

    # Deaths -> Thousands
    fig.update_yaxes(tickformat=".2s", row=1, col=2)

    # Active Cases -> Lakhs
    fig.update_yaxes(tickformat=".2s", row=2, col=1)

    # CFR -> %
    fig.update_yaxes(ticksuffix="%", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)


    rolling = merged['NewConfirmed_7d']
    def find_peaks(s, min_h, min_dist=60):
        vals, peaks = s.values, []
        for i in range(1, len(vals)-1):
            if vals[i]>vals[i-1] and vals[i]>vals[i+1] and vals[i]>=min_h:
                if not peaks or (i - peaks[-1][0]) >= min_dist:
                    peaks.append((i, s.index[i], vals[i]))
        return peaks
    hist_peaks = find_peaks(rolling, min_h=50_000, min_dist=60)


    if len(hist_peaks) >= 2:
        wave1 = hist_peaks[0][2]
        wave2 = hist_peaks[1][2]
        severity_ratio = wave2 / wave1
        peak_date = hist_peaks[1][1]

        c1, c2, c3 = st.columns(3)

        c1.metric("Peak Cases (Wave 2)", f"{wave2:,.0f}")
        c2.metric("Severity Ratio", f"{severity_ratio:.2f}x")
        c3.metric("Peak Date", peak_date.strftime("%d %b %Y"))

    fig = go.Figure()

    # Area (fill)
    fig.add_trace(go.Scatter(
        x=merged.index,
        y=rolling,
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.15)',
        line=dict(color='rgba(255,0,0,0)'),
        hoverinfo='skip',
        showlegend=False
    ))

    # Line
    fig.add_trace(go.Scatter(
        x=merged.index,
        y=rolling,
        name='7-day rolling avg',
        line=dict(color='red', width=3)
    ))
    labels = ['Wave 1', 'Wave 2']

    for i, (_, date, val) in enumerate(hist_peaks):
        lbl = labels[i] if i < len(labels) else f'Wave {i+1}'

        # Vertical line
        fig.add_vline(
            x=date,
            line_dash="dash",
            line_color="teal",
            opacity=0.6
        )

        # Annotation
        fig.add_annotation(
            x=date,
            y=val,
            text=f"{lbl}<br>{date.strftime('%b %Y')}<br>{val/1e3:.0f}K/day",
            showarrow=True,
            arrowhead=2,
            ax=40,   # horizontal shift
            ay=-40,  # vertical shift
            font=dict(size=11, color="teal")
        )
        vacc_start = pd.Timestamp('2021-03-08')

    fig.add_vline(
        x=vacc_start,
        line_dash="dot",
        line_color="green",
        line_width=2
    )

    fig.add_annotation(
        x=vacc_start,
        y=rolling.max()*0.6,
        text="Vaccination<br>data starts",
        showarrow=False,
        font=dict(color="green")
    )
    fig.update_layout(
        title="India COVID-19 — Wave Identification with Vaccination Milestone",
        xaxis_title="Date",
        yaxis_title="New Cases / Day (7d avg)",
        template="plotly_white",
        height=500
    )
    fig.update_yaxes(
        tickformat=".2s"  # auto K, M formatting
    )
    fig.update_traces(
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Cases: %{y:,.0f}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Wave Insights")
    col1, col2, col3= st.columns(3)
    if len(hist_peaks) >= 2:
        st.info(f"""
        **Wave Comparison:**  
        Wave 2 peaked at **{wave2:,.0f} cases/day**, which is  
        **{severity_ratio:.1f}x higher** than Wave 1.

        This indicates a significantly more aggressive spread,
        likely due to higher transmissibility and delayed interventions.
        """)
    cfr_mean = merged['CFR_pct'].mean()
    cfr_latest = merged['CFR_pct'].iloc[-1]
    with col1:
        st.metric("CFR Change", f"{cfr_latest - cfr_mean:.2f}%")

    st.info("Declining CFR indicates improved treatment and healthcare response.")
    if len(hist_peaks) >= 2:
        wave1 = hist_peaks[0][2]
        wave2 = hist_peaks[1][2]
    else:
        st.warning("Not enough peaks detected for wave comparison.")

    severity_ratio = wave2 / wave1
    with col2:
        st.metric("Wave 2 vs Wave 1 Severity", f"{severity_ratio:.2f}x")

    st.info(
        f"Wave 2 peak was {severity_ratio:.1f} times higher than Wave 1, "
        "indicating a significantly more aggressive spread."
    )
    peak_date = hist_peaks[1][1]

    post_peak = merged.loc[peak_date:]["NewConfirmed_7d"]

    half_mask = post_peak < post_peak.max() * 0.5

    if half_mask.any():
        decline_days = half_mask.idxmax()
        days_to_half = (decline_days - peak_date).days
    else:
        days_to_half = None
    with col3:
        if days_to_half:
            st.metric("Days to Reduce 50%", f"{days_to_half} days")

    st.markdown('#### Historical peaks:')
    for i, (_, d, v) in enumerate(hist_peaks):
        st.info(f'  Wave {i+1}: {d.strftime("%d %B %Y")}  →  {v:,.0f} cases/day')
    st.info(f'  Inter-wave gap: {(hist_peaks[1][1]-hist_peaks[0][1]).days} days')

    st.markdown("---")

    st.markdown("## Vaccination Impact Analysis")
    st.write('This section analyzes the impact of COVID-19 vaccination efforts on the trends of new cases and deaths in India. By comparing the timelines of vaccination rollout with the peaks and declines in cases, we can gain insights into how vaccination may have influenced the course of the pandemic.')
    vacc_zone = merged[merged['pct_dose1'] > 0]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged['NewConfirmed_7d'],
            name="COVID new cases (7d avg)",
            line=dict(color='red', width=3),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.15)'
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=vacc_zone.index,
            y=vacc_zone['daily_doses_7d'] / 1e6,
            name="Daily doses (M)",
            line=dict(color='green'),
            fill='tozeroy',
            fillcolor='rgba(0,200,0,0.25)'
        ),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=vacc_zone.index,
            y=vacc_zone['pct_dose1'],
            name="Dose-1 (%)",
            line=dict(color='blue', dash='dash', width=2)
        ),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=vacc_zone.index,
            y=vacc_zone['pct_dose2'],
            name="Dose-2 (%)",
            line=dict(color='purple', dash='dot', width=2)
        ),
        secondary_y=True
    )
    peak_date = hist_peaks[1][1]
    peak_val  = hist_peaks[1][2]

    fig.add_annotation(
        x=peak_date,
        y=peak_val,
        text="Wave 2 peak<br>(pre-vaccination mass rollout)",
        showarrow=True,
        arrowhead=2,
        ax=-80,
        ay=-60,
        font=dict(color='red')
    )
    last_vacc = vacc_zone.iloc[-1]

    fig.add_annotation(
        x=vacc_zone.index[-1],
        y=last_vacc['pct_dose1'],
        text=f"Dose-1: {last_vacc['pct_dose1']:.1f}%",
        showarrow=True,
        arrowhead=2,
        ax=-60,
        ay=-40,
        font=dict(color='blue')
    )
    fig.update_layout(
        title="COVID-19 Cases vs Vaccination Rollout — India (2020-2021)<br>"
            "<sup>Inverse relationship emerges as vaccination coverage increases</sup>",
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", y=1.05)
    )
    fig.update_yaxes(
        title_text="Daily New Confirmed Cases (7d avg)",
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="Vaccination Coverage (%) / Daily Doses (Millions)",
        secondary_y=True
    )
    fig.update_traces(
        hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:,.2f}<extra></extra>"
    )
    fig.add_vrect(
        x0=merged.index.min(),
        x1=vacc_zone.index.min(),
        fillcolor="gray",
        opacity=0.1,
        layer="below",
        line_width=0,
        annotation_text="Pre-vaccination",
        annotation_position="top left"
    )
    fig.update_xaxes(
        tickformat="%b %Y"
    )
    st.plotly_chart(fig, use_container_width=True)
    merged=merged[merged['pct_dose1'] > 10]
    corr = merged['pct_dose1'].corr(merged['NewConfirmed_7d'])
    
    st.metric("Vaccination vs Cases Correlation", f"{corr:.2f}")

    st.info(
    "After vaccination rollout, a negative correlation is observed, suggesting that higher vaccination coverage is associated with lower case trends. However, correlation does not imply causation."
)
    st.markdown("###")
    st.markdown("---")
