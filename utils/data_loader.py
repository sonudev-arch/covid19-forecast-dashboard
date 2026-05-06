import pandas as pd
import streamlit as st

@st.cache_data
def load_covid_data():
    return pd.read_csv('datasets/covid_19_india.csv')

@st.cache_data
def load_vaccination_data():
    return pd.read_csv('datasets/govdatavacc.csv')

@st.cache_data
def load_statewise_data():
    return pd.read_csv('datasets/statewise.csv')

@st.cache_data
def load_vaccination2_data():
    return pd.read_csv('datasets/vaccination.csv')

@st.cache_data
def load_merged_data(): 
    return pd.read_csv('datasets/merged_covid_vaccination.csv')

@st.cache_data
def load_merged(): 
    return pd.read_csv('datasets/merged_covid_vacc.csv')