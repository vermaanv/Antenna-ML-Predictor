# ============================================================
# Page 1 — Antenna Performance Predictor
# ============================================================

import os
import sys
import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_loader import load_models, load_data

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Antenna Predictor",
    page_icon="🎯",
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

# ── Load Models ───────────────────────────────────────────────
xgb_freq, rf_s11, rf_class, scaler, ann_s11, ann_freq = load_models()

# ── Page Content ──────────────────────────────────────────────
st.title("🎯 Antenna Performance Predictor")
st.markdown("Enter antenna dimensions to predict performance instantly.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📐 Antenna Dimensions")
    PL  = st.slider("Patch Length - PL (mm)",
                    min_value=26.0, max_value=33.0,
                    value=29.45, step=0.1)
    PW  = st.slider("Patch Width - PW (mm)",
                    min_value=34.0, max_value=43.0,
                    value=38.39, step=0.1)
    INL = st.slider("Inset Notch Length - INL (mm)",
                    min_value=6.0, max_value=10.0,
                    value=7.84, step=0.1)
    ML  = st.slider("Feed Line Length - ML (mm)",
                    min_value=14.0, max_value=20.0,
                    value=17.29, step=0.1)

with col2:
    st.subheader("📊 Predicted Performance")

    if st.button("🔮 Predict Now", use_container_width=True):

        input_data        = np.array([[PL, PW, INL, ML]])
        input_data_scaled = scaler.transform(input_data)

        # Traditional ML predictions
        freq_pred  = xgb_freq.predict(input_data)[0]
        s11_pred   = rf_s11.predict(input_data)[0]
        class_pred = rf_class.predict(input_data)[0]

        # ANN predictions
        ann_s11_pred  = ann_s11.predict(input_data_scaled, verbose=0)[0][0]
        ann_freq_pred = ann_freq.predict(input_data_scaled, verbose=0)[0][0]

        st.markdown("### Results:")

        # Traditional ML
        st.markdown("**Traditional ML (XGBoost + Random Forest):**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                label="Resonant Frequency (XGBoost)",
                value=f"{freq_pred:.3f} GHz",
                delta=f"{freq_pred - 2.4:.3f} GHz from 2.4 GHz"
            )
        with col_b:
            st.metric(
                label="Predicted S11 (Random Forest)",
                value=f"{s11_pred:.2f} dB"
            )

        # ANN
        st.markdown("**Neural Network (ANN):**")
        col_c, col_d = st.columns(2)
        with col_c:
            st.metric(
                label="Resonant Frequency (ANN)",
                value=f"{ann_freq_pred:.3f} GHz",
                delta=f"{ann_freq_pred - 2.4:.3f} GHz from 2.4 GHz"
            )
        with col_d:
            st.metric(
                label="Predicted S11 (ANN)",
                value=f"{ann_s11_pred:.2f} dB"
            )

        if class_pred == 1:
            st.success("✅ GOOD ANTENNA — S11 < -10 dB")
        else:
            st.error("❌ POOR ANTENNA — S11 ≥ -10 dB")

        st.markdown("---")

        # S11 Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=s11_pred,
            title={'text': "S11 Value (dB)"},
            gauge={
                'axis': {'range': [-50, 0]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-50, -20], 'color': "green"},
                    {'range': [-20, -10], 'color': "yellow"},
                    {'range': [-10, 0],   'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': -10
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Frequency Gauge
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=freq_pred,
            title={'text': "Resonant Frequency (GHz)"},
            delta={'reference': 2.4},
            gauge={
                'axis': {'range': [2.0, 3.0]},
                'bar': {'color': "royalblue"},
                'steps': [
                    {'range': [2.0, 2.3], 'color': "lightcoral"},
                    {'range': [2.3, 2.5], 'color': "lightgreen"},
                    {'range': [2.5, 3.0], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 2.4
                }
            }
        ))
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("👈 Adjust sliders and click Predict Now")