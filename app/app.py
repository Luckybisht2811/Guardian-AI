import os
import requests
import numpy as np
import pandas as pd
import pickle
import folium
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import DBSCAN
from folium import plugins
from streamlit_folium import folium_static
import streamlit.components.v1 as components

# Import custom modules
from Criminal_Profiling import create_criminal_profiling_dashboard
from Crime_Pattern_Analysis import *
from Predictive_modeling import *
from Resource_Allocation import *
from Continuous_Learning_and_Feedback import *

# -----------------------------------------------
# Custom CSS for Styling
# -----------------------------------------------
st.markdown("""
<style>
/* Background and Font */
body {
    background-color: #0d1117;
    color: #e5e5e5;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar design */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    color: white;
    border-right: 2px solid #62d0ff;
}

/* Title Styling */
h1, h2, h3, h4 {
    color: #62d0ff;
}

/* Button Styling */
.stButton button {
    background-color: #62d0ff;
    color: #0d1117;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    transition: 0.3s;
}
.stButton button:hover {
    background-color: #0d1117;
    color: #62d0ff;
    border: 1px solid #62d0ff;
}

/* Container and cards */
.block-container {
    background: rgba(255,255,255,0.02);
    padding: 2rem;
    border-radius: 10px;
}

/* Center title */
.title-center {
    text-align: center;
    font-size: 40px;
    color: #62d0ff;
    text-shadow: 0 0 10px #62d0ff;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# Sidebar Menu
# -----------------------------------------------
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

with st.sidebar:
    selected = option_menu(
        "GuardianAI",
        [
            'Home',
            'Crime Pattern Analysis',
            "Criminal Profiling",
            'Predictive Modeling',
            'Police Resource Allocation',
            'Continuous Learning and Feedback',
            'Documentation and Resources'
        ],
        icons=[
            'house-fill',
            'bar-chart-fill',
            "fingerprint",
            'cpu-fill',
            'diagram-3-fill',
            'book-fill',
            'file-earmark-text-fill'
        ],
        menu_icon="shield-shaded",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5!important", "background-color": "#0f2027"},
            "menu-title": {"font-size": "18px", "font-weight": "bold", "color": "#e5e5e5"},
            "menu-icon": {"color": "#62d0ff"},
            "nav": {"background-color": "#0f2027"},
            "nav-item": {"padding": "0px 10px"},
            "nav-link": {
                "text-decoration": "none",
                "color": "#e5e5e5",
                "font-size": "14px",
                "font-weight": "normal",
                "--hover-color": "#62d0ff",
            },
            "nav-link-selected": {
                "background-color": "#62d0ff",
                "color": "#0f2027",
                "font-weight": "bold",
            },
            "icon": {"color": "#e5e5e5", "font-size": "16px"},
        }
    )

# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# HOME PAGE
# -----------------------------------------------
if selected == "Home":
    st.markdown("<h1 class='title-center'>🛡️ GuardianAI: Smart Crime Prediction System</h1>", unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### 🔍 Overview
            **GuardianAI** is an intelligent, AI-driven system designed to revolutionize crime prevention and law enforcement efficiency.

            Using **data analytics**, **machine learning**, and **predictive modeling**, GuardianAI enables agencies to:
            - Detect crime hotspots
            - Predict future crime trends
            - Optimize police resource allocation
            - Profile repeat offenders effectively
            """
        )

        st.markdown(
            """
            ### 🚔 Key Features
            - **Crime Pattern Analysis** – Identify spatial & temporal crime patterns.
            - **Criminal Profiling** – Understand offender behavior for better prevention.
            - **Predictive Modeling** – Forecast where and when crimes might occur.
            - **Resource Allocation** – Deploy forces efficiently to high-risk areas.
            - **Continuous Learning** – AI system adapts and evolves using new feedback.
            """
        )

        if st.button("📘 Learn More"):
            st.session_state.selected_page = "Documentation and Resources"
            st.experimental_rerun()

    with col2:
        data_file_path = os.path.join(root_dir, 'assets', 'Home_Page_image.jpg')
        st.image(data_file_path, use_container_width=True)


# -----------------------------------------------
# CRIME PATTERN ANALYSIS
# -----------------------------------------------
if selected == "Crime Pattern Analysis":

    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/adarshbiradar/maps-geojson/master/states/karnataka.json"
        response = requests.get(url)
        geojson_data = response.json()
        data_file_path = os.path.join(root_dir, 'Component_datasets', 'Crime_Pattern_Analysis_Cleaned.csv')
        df = pd.read_csv(data_file_path)
        mean_lat = df['Latitude'].mean()
        mean_lon = df['Longitude'].mean()
        return mean_lat, mean_lon, geojson_data, df

    mean_lat, mean_lon, geojson_data, df = load_data()

    st.subheader("📆 Temporal Analysis of Crime Data")
    temporal_analysis(df)

    st.subheader("🗺️ Choropleth Crime Maps")
    chloropleth_maps(df, geojson_data, mean_lat, mean_lon)

    st.subheader("🔥 Crime Hotspot Detection")
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    crime_hotspots(df, mean_lat, mean_lon)

# -----------------------------------------------
# CRIMINAL PROFILING
# -----------------------------------------------
if selected == "Criminal Profiling":
    create_criminal_profiling_dashboard()

# -----------------------------------------------
# PREDICTIVE MODELING
# -----------------------------------------------
if selected == "Predictive Modeling":
    predictive_modeling_recidivism()

# -----------------------------------------------
# RESOURCE ALLOCATION
# -----------------------------------------------
if selected == "Police Resource Allocation":
    data_file_path = os.path.join(root_dir, 'Component_datasets', 'Resource_Allocation_Cleaned.csv')
    df = pd.read_csv(data_file_path)
    resource_allocation(df)

# -----------------------------------------------
# CONTINUOUS LEARNING
# -----------------------------------------------
if selected == "Continuous Learning and Feedback":
    continuous_learning_and_feedback()

# -----------------------------------------------
# DOCUMENTATION
# -----------------------------------------------
if selected == "Documentation and Resources":
    st.markdown('Click [here](hhttps://github.com/Luckybisht2811/Guardian-AI) to view the documentation and resources.')
