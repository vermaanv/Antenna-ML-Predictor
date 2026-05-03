# ============================================================
# Antenna ML Predictor - Main Entry Point
# Project: ML-Assisted Microstrip Patch Antenna Design
# ============================================================

import streamlit as st

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Antenna ML Predictor",
    page_icon="📡",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("📡 Antenna ML Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("**Antenna Specs:**")
st.sidebar.markdown("- Type: Microstrip Patch")
st.sidebar.markdown("- Frequency: 2.4 GHz")
st.sidebar.markdown("- Substrate: FR4 (εr=4.3)")
st.sidebar.markdown("- S11: -31.77 dB")
st.sidebar.markdown("- Gain: 6.15 dBi")

# ── Home Page ─────────────────────────────────────────────────
st.title("📡 Antenna ML Predictor")
st.markdown("### ML-Assisted Microstrip Patch Antenna Design System")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**🎯 Antenna Predictor**\nPredict frequency and S11 from antenna dimensions instantly")
with col2:
    st.info("**🔧 Antenna Optimizer**\nFind optimal dimensions for any target frequency")
with col3:
    st.info("**📡 Antenna Design**\nView CST simulation results and antenna specifications")

st.markdown("---")

col4, col5 = st.columns(2)
with col4:
    st.success("**📊 Model Performance**\nCompare all 4 ML models — Traditional ML vs ANN")
with col5:
    st.success("**🗄️ Dataset Explorer**\nExplore the 260 sample CST simulation dataset")

st.markdown("---")

# ── Project Summary ───────────────────────────────────────────
st.subheader("📋 Project Summary")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("CST Simulations", "400")
with col2:
    st.metric("Clean Samples", "260")
with col3:
    st.metric("Freq Accuracy", "99.94%")
with col4:
    st.metric("S11 Classification", "96.15%")

st.markdown("---")

st.subheader("🏆 Key Results")

results_data = {
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'ANN (Neural Network)'],
    'Frequency R²': [0.9934, 0.9988, 0.9994, 0.9826],
    'S11 R²': [0.3078, 0.3421, 0.0649, 0.7010],
    'Best For': ['Baseline', 'S11 Classification', 'Frequency Prediction', 'S11 Prediction']
}

import pandas as pd
results_df = pd.DataFrame(results_data)
st.dataframe(results_df, use_container_width=True)

st.markdown("---")
st.markdown("**👈 Use the sidebar to navigate between pages**")