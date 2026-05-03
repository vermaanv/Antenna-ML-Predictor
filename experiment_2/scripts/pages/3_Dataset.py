# ============================================================
# Page 3 — Dataset Explorer
# ============================================================

import os
import sys
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_loader import load_data

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="🗄️",
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

# ── Load Data ─────────────────────────────────────────────────
df = load_data()

# ── Page Content ──────────────────────────────────────────────
st.title("🗄️ Dataset Explorer")
st.markdown("---")

# ── Metrics ───────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Samples", "260")
with col2:
    st.metric("Parameters", "4")
with col3:
    st.metric("CST Simulations", "400")

st.markdown("---")

# ── Dataset Preview ───────────────────────────────────────────
st.subheader("📋 Dataset Preview")
st.dataframe(df, use_container_width=True)

st.markdown("---")

# ── Parameter Distributions ───────────────────────────────────
st.subheader("📊 Parameter Distributions")
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x='PL',
                      title='Patch Length Distribution',
                      color_discrete_sequence=['blue'])
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(df, x='INL',
                       title='Inset Notch Length Distribution',
                       color_discrete_sequence=['green'])
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = px.histogram(df, x='PW',
                       title='Patch Width Distribution',
                       color_discrete_sequence=['orange'])
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(df, x='ML',
                       title='Feed Line Length Distribution',
                       color_discrete_sequence=['red'])
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── S11 vs Frequency ──────────────────────────────────────────
st.subheader("📡 S11 vs Resonant Frequency")
fig5 = px.scatter(
    df, x='Frequency', y='S11',
    color='ML',
    title='S11 vs Resonant Frequency (colored by Feed Length)',
    labels={
        'Frequency': 'Resonant Frequency (GHz)',
        'S11'      : 'S11 (dB)',
        'ML'       : 'Feed Length (mm)'
    }
)
fig5.add_hline(y=-10, line_dash="dash",
               line_color="red",
               annotation_text="-10 dB threshold")
st.plotly_chart(fig5, use_container_width=True)