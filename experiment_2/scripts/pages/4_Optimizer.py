# ============================================================
# Page 4 — Antenna Inverse Design Optimizer
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import differential_evolution

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_loader import load_models

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Antenna Optimizer",
    page_icon="🔧",
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
st.title("🔧 Antenna Inverse Design Optimizer")
st.markdown("Enter your **desired frequency** — ML will find optimal antenna dimensions instantly.")
st.markdown("---")

# ── How it works ──────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Step 1**\nEnter your target frequency")
with col2:
    st.info("**Step 2**\nClick Optimize")
with col3:
    st.info("**Step 3**\nGet optimal dimensions instantly")

st.markdown("---")

# ── Input & Output ────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Target Performance")
    target_freq = st.slider(
        "Target Frequency (GHz)",
        min_value=2.1,
        max_value=2.7,
        value=2.4,
        step=0.01
    )
    st.markdown(f"**Target:** {target_freq} GHz")
    st.markdown("**Constraint:** S11 < -10 dB (Good antenna)")
    st.markdown("**Method:** Differential Evolution Optimization")

with col2:
    st.subheader("⚙️ Optimization Result")

    if st.button("🚀 Optimize Now", use_container_width=True):

        # ── Objective function ────────────────────────────────
        def objective(params):
            PL, PW, INL, ML = params
            input_data  = np.array([[PL, PW, INL, ML]])
            freq_pred   = xgb_freq.predict(input_data)[0]
            s11_pred    = rf_s11.predict(input_data)[0]
            freq_error  = abs(freq_pred - target_freq)
            s11_penalty = max(0, s11_pred + 10) * 0.1
            return freq_error + s11_penalty

        # ── Run optimization ──────────────────────────────────
        BOUNDS = [
            (26.0, 33.0),
            (34.0, 43.0),
            (6.0,  10.0),
            (14.0, 20.0)
        ]

        with st.spinner('🔍 Optimizing antenna dimensions...'):
            result = differential_evolution(
                objective,
                bounds=BOUNDS,
                maxiter=1000,
                tol=1e-6,
                seed=42,
                polish=True
            )

        # ── Get optimal dimensions ────────────────────────────
        PL_opt, PW_opt, INL_opt, ML_opt = result.x
        input_opt      = np.array([[PL_opt, PW_opt, INL_opt, ML_opt]])
        freq_pred      = xgb_freq.predict(input_opt)[0]
        s11_pred       = rf_s11.predict(input_opt)[0]
        class_pred     = rf_class.predict(input_opt)[0]
        freq_error     = abs(freq_pred - target_freq)
        freq_error_pct = (freq_error / target_freq) * 100

        # ── Display results ───────────────────────────────────
        st.markdown("### ✅ Optimal Dimensions Found!")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Patch Length (PL)", f"{PL_opt:.2f} mm")
            st.metric("Inset Length (INL)", f"{INL_opt:.2f} mm")
        with col_b:
            st.metric("Patch Width (PW)", f"{PW_opt:.2f} mm")
            st.metric("Feed Length (ML)", f"{ML_opt:.2f} mm")

        st.markdown("---")
        st.markdown("### 📊 Predicted Performance:")

        col_c, col_d = st.columns(2)
        with col_c:
            st.metric(
                label="Predicted Frequency",
                value=f"{freq_pred:.4f} GHz",
                delta=f"{freq_error_pct:.3f}% error"
            )
        with col_d:
            st.metric(
                label="Predicted S11",
                value=f"{s11_pred:.2f} dB"
            )

        if class_pred == 1:
            st.success("✅ GOOD ANTENNA — S11 < -10 dB")
        else:
            st.warning("⚠️ POOR ANTENNA — Consider adjusting target frequency")

        st.markdown("---")

        # ── Frequency gauge ───────────────────────────────────
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=freq_pred,
            title={'text': "Predicted Frequency (GHz)"},
            delta={'reference': target_freq},
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
                    'value': target_freq
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # ── S11 gauge ─────────────────────────────────────────
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=s11_pred,
            title={'text': "Predicted S11 (dB)"},
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
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("👈 Set target frequency and click Optimize Now")

st.markdown("---")

# ── Pre-computed results table ────────────────────────────────
st.subheader("📋 Pre-computed Optimization Results")
st.markdown("Results for common target frequencies:")

opt_data = {
    'Target (GHz)': [2.30, 2.35, 2.40, 2.45, 2.50],
    'PL (mm)':  [31.27, 31.12, 30.27, 29.87, 29.06],
    'PW (mm)':  [34.18, 34.12, 40.35, 35.91, 42.60],
    'INL (mm)': [7.49,  7.16,  8.01,  6.97,  6.63],
    'ML (mm)':  [18.47, 15.11, 19.80, 19.59, 14.85],
    'Predicted (GHz)': [2.2907, 2.3520, 2.3999, 2.4294, 2.5056],
    'S11 (dB)': [-10.58, -10.01, -18.19, -24.27, -18.41],
    'Error (%)': [0.404, 0.086, 0.004, 0.841, 0.225],
    'Quality': ['GOOD ✅', 'POOR ❌', 'GOOD ✅', 'GOOD ✅', 'GOOD ✅']
}

opt_df = pd.DataFrame(opt_data)
st.dataframe(opt_df, use_container_width=True)