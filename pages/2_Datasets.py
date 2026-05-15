import streamlit as st
import pandas as pd

from utils.data_loader import (
    load_covid_data,
    load_vaccination_data,
    load_vaccination2_data
)
@st.cache_data
def load_covid_cached():
    return load_covid_data()

@st.cache_data
def load_vacc_cached():
    return load_vaccination_data()

@st.cache_data
def load_vacc2_cached():
    return load_vaccination2_data()

with st.spinner("Loading datasets..."):
    covid_data = load_covid_cached()
    vac_data   = load_vacc_cached()
    vac2_data  = load_vacc2_cached()

def show_dataset(title, data, link):

    st.header(title)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", data.shape[0])
    col2.metric("Columns", data.shape[1])
    col3.metric("Missing Values", data.isnull().sum().sum())

    show_date_range(data)

    st.subheader("Data Preview")

    key_prefix = title.replace(" ", "_")

    if f"{key_prefix}_rows" not in st.session_state:
        st.session_state[f"{key_prefix}_rows"] = 10

    def update_slider():
        st.session_state[f"{key_prefix}_rows"] = st.session_state[f"{key_prefix}_slider"]

    def update_input():
        st.session_state[f"{key_prefix}_rows"] = st.session_state[f"{key_prefix}_input"]

    col1, col2 = st.columns(2)

    with col1:
        st.slider(
            "Select rows",
            5,
            len(data),
            value=st.session_state[f"{key_prefix}_rows"],
            key=f"{key_prefix}_slider",
            on_change=update_slider
        )

    with col2:
        st.number_input(
            "Enter rows",
            5,
            len(data),
            value=st.session_state[f"{key_prefix}_rows"],
            key=f"{key_prefix}_input",
            on_change=update_input
        )

    st.dataframe(data.head(st.session_state[f"{key_prefix}_rows"]))

    st.subheader("Dataset Info")

    info_df = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str),
        "Non-Null Count": data.count().values
    })

    st.dataframe(info_df)

    st.subheader("Data Quality")

    col1, col2 = st.columns(2)

    col1.write("Missing Values")
    col1.dataframe(data.isnull().sum())

    col2.write("Duplicate Rows")
    col2.write(data.duplicated().sum())

    st.subheader("Statistical Summary")
    st.write(data.describe())

    st.link_button("View Dataset", link)

    st.markdown("---")
tabs=st.tabs(["Overview",
              "Covid 19(India)",
              "Vaccination Data(Kaggle)",
              "Vaccination Data(Datagov)"])
with tabs[0]:
    st.title("Datasets")
    st.markdown("## Overview")

    st.write("""
    This section presents the datasets used for analyzing COVID-19 trends in India.

    The project combines multiple time-series datasets covering case statistics
    and vaccination progress to provide a comprehensive view of the pandemic.

    These datasets enable analysis of:
    - Pandemic progression over time  
    - Regional spread across states and union territories  
    - Impact of vaccination on case trends  
    - Wave patterns and growth dynamics  

    They form the foundation for all analytical tasks in this system,
    including trend analysis, wave detection, and forecasting.
    """)

    st.markdown("---")
#Function to show date range coverage
def show_date_range(df, possible_cols=["Date", "date", "Updated On", "Time", "Timestamp"]):
    for col in possible_cols:
        if col in df.columns:
            try:
                temp = pd.to_datetime(df[col], errors="coerce")
                min_date = temp.min()
                max_date = temp.max()
                

                st.subheader("Date Range Coverage")

                col1, col2 = st.columns(2)
                col1.metric("Start Date", str(min_date.date()) if pd.notnull(min_date) else "N/A")
                col2.metric("End Date", str(max_date.date()) if pd.notnull(max_date) else "N/A")

                st.caption(f":blue[Total Duration]: {(max_date - min_date).days} days" if pd.notnull(min_date) and pd.notnull(max_date) else "")
                return
            except:
                continue

    st.subheader("Date Range Coverage")
    st.warning("No valid date column found in dataset.")


with tabs[1]:
    show_dataset(
        "COVID-19 Data (India)",
        covid_data,
        "https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india"
    )


with tabs[2]:
    show_dataset(
        "Vaccination Data (Kaggle)",
        vac2_data,
        "https://www.kaggle.com/datasets/harveenchadha/india-covid19-vaccination-data"
    )

with tabs[3]:
        
    show_dataset(
        "Vaccination Data (Datagov)",
        vac_data,
        "https://www.data.gov.in/resource/stateut-wise-covid-19-vaccination-status-5-august-2021"
    )
    st.subheader("Preprocessing Summary")

    st.write("""
    - Converted date columns to datetime  
    - Handled missing values  
    - Sorted data chronologically  
    - Prepared features for time-series modeling  
    """)
    cols = st.columns(6)
    with cols[0]:
        if st.button("Back"):
            st.switch_page("Home.py")
    with cols[5]:
        if st.button("Next"):
            st.switch_page("pages/3_Data_Analysis.py")

    st.markdown("---")