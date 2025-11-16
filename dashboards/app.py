import streamlit as st
from src.visualization import PortfolioVisualizer

st.title("Asset Management Dashboard")
visualizer = PortfolioVisualizer(portfolio)
st.plotly_chart(visualizer.dashboard(), use_container_width=True)