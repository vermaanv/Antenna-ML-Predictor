# ============================================================
# Antenna ML Predictor - Web Application
# Project: ML-Assisted Microstrip Patch Antenna Design
# Description: Streamlit web app for antenna performance
#              prediction using trained ML models
# ============================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.optimize import differential_evolution

warnings.filterwarnings('ignore')

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Antenna ML Predictor",
    page_icon="📡",
    layout="wide"
)

# ── Base Directory ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load Trained Models ───────────────────────────────────────
@st.cache_resource
def load_models():
    with open(os.path.join(BASE_DIR, '../models/xgb_freq.pkl'), 'rb') as f:
        xgb_freq = pickle.load(f)
    with open(os.path.join(BASE_DIR, '../models/rf_s11.pkl'), 'rb') as f:
        rf_s11 = pickle.load(f)
    with open(os.path.join(BASE_DIR, '../models/rf_class.pkl'), 'rb') as f:
        rf_class = pickle.load(f)
    return xgb_freq, rf_s11, rf_class

# ── Load Dataset ──────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, '../data/antenna_dataset_400.csv'))

xgb_freq, rf_s11, rf_class = load_models()
df = load_data()

# ── Sidebar Navigation ────────────────────────────────────────
st.sidebar.title("📡 Antenna ML Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🎯 Antenna Predictor",
     "📊 Model Performance",
     "🗄️ Dataset Explorer",
     "🔧 Antenna Optimizer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Antenna Specs:**")
st.sidebar.markdown("- Type: Microstrip Patch")
st.sidebar.markdown("- Frequency: 2.4 GHz")
st.sidebar.markdown("- Substrate: FR4 (εr=4.3)")
st.sidebar.markdown("- S11: -31.77 dB")
st.sidebar.markdown("- Gain: 6.15 dBi")

# ════════════════════════════════════════════════════════════
# PAGE 1 — ANTENNA PREDICTOR
# ════════════════════════════════════════════════════════════
if page == "🎯 Antenna Predictor":

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

            input_data = np.array([[PL, PW, INL, ML]])

            freq_pred  = xgb_freq.predict(input_data)[0]
            s11_pred   = rf_s11.predict(input_data)[0]
            class_pred = rf_class.predict(input_data)[0]

            st.markdown("### Results:")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="Resonant Frequency",
                    value=f"{freq_pred:.3f} GHz",
                    delta=f"{freq_pred - 2.4:.3f} GHz from 2.4 GHz"
                )
            with col_b:
                st.metric(
                    label="Predicted S11",
                    value=f"{s11_pred:.2f} dB"
                )

            if class_pred == 1:
                st.success("✅ GOOD ANTENNA — S11 < -10 dB")
            else:
                st.error("❌ POOR ANTENNA — S11 ≥ -10 dB")

            st.markdown("---")

            # S11 Gauge chart
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

            # Frequency Gauge chart
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

# ════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.title("📊 Model Performance")
    st.markdown("---")

    # ── Experiment comparison table ───────────────────────────
    st.subheader("🔬 Experiment Comparison")
    comp_data = {
        'Experiment': ['Experiment 1', 'Experiment 2'],
        'Parameters': ['PL, PW, INL (3)', 'PL, PW, INL, ML (4)'],
        'Samples': [74, 260],
        'Best Freq R²': [0.9643, 0.9994],
        'Best S11 R²': [0.3223, 0.3421],
        'S11 Classification': ['N/A', '96.15%']
    }
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True)
    st.markdown("---")

    # ── Prepare data for plots ────────────────────────────────
    X      = df[['PL', 'PW', 'INL', 'ML']]
    y_freq = df['Frequency']
    y_s11  = df['S11']

    X_train, X_test, yf_train, yf_test = train_test_split(
        X, y_freq, test_size=0.2, random_state=42)
    _, _, ys_train, ys_test = train_test_split(
        X, y_s11, test_size=0.2, random_state=42)

    # ── Train all models ──────────────────────────────────────
    models_freq = {
        'Linear Regression': LinearRegression(),
        'Random Forest'    : RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost'          : XGBRegressor(n_estimators=100, random_state=42)
    }
    models_s11 = {
        'Linear Regression': LinearRegression(),
        'Random Forest'    : RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost'          : XGBRegressor(n_estimators=100, random_state=42)
    }

    freq_preds = {}
    for name, model in models_freq.items():
        model.fit(X_train, yf_train)
        freq_preds[name] = {
            'pred': model.predict(X_test),
            'r2'  : r2_score(yf_test, model.predict(X_test)),
            'mae' : mean_absolute_error(yf_test, model.predict(X_test))
        }

    s11_preds = {}
    for name, model in models_s11.items():
        model.fit(X_train, ys_train)
        s11_preds[name] = {
            'pred': model.predict(X_test),
            'r2'  : r2_score(ys_test, model.predict(X_test)),
            'mae' : mean_absolute_error(ys_test, model.predict(X_test))
        }

    # ── R² bar charts ─────────────────────────────────────────
    st.subheader("📈 R² Score Comparison")
    col1, col2 = st.columns(2)

    with col1:
        r2_freq_vals = [v['r2'] for v in freq_preds.values()]
        fig = px.bar(
            x=list(freq_preds.keys()),
            y=r2_freq_vals,
            title='Frequency Prediction R² (Experiment 2)',
            labels={'x': 'Model', 'y': 'R² Score'},
            color=r2_freq_vals,
            color_continuous_scale='Greens'
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        r2_s11_vals = [v['r2'] for v in s11_preds.values()]
        fig2 = px.bar(
            x=list(s11_preds.keys()),
            y=r2_s11_vals,
            title='S11 Prediction R² (Experiment 2)',
            labels={'x': 'Model', 'y': 'R² Score'},
            color=r2_s11_vals,
            color_continuous_scale='Blues'
        )
        fig2.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Frequency Actual vs Predicted — all 3 models ──────────
    st.subheader("🎯 Frequency Prediction — Actual vs Predicted")
    col1, col2, col3 = st.columns(3)

    for idx, (name, res) in enumerate(freq_preds.items()):
        with [col1, col2, col3][idx]:
            fig = px.scatter(
                x=yf_test, y=res['pred'],
                labels={'x': 'Actual (GHz)', 'y': 'Predicted (GHz)'},
                title=f'{name}\nR²={res["r2"]:.4f}'
            )
            fig.add_shape(
                type='line',
                x0=yf_test.min(), y0=yf_test.min(),
                x1=yf_test.max(), y1=yf_test.max(),
                line=dict(color='red', dash='dash')
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── S11 Actual vs Predicted — all 3 models ────────────────
    st.subheader("📉 S11 Prediction — Actual vs Predicted")
    col1, col2, col3 = st.columns(3)

    for idx, (name, res) in enumerate(s11_preds.items()):
        with [col1, col2, col3][idx]:
            fig = px.scatter(
                x=ys_test, y=res['pred'],
                labels={'x': 'Actual S11 (dB)', 'y': 'Predicted S11 (dB)'},
                title=f'{name}\nR²={res["r2"]:.4f}',
                color_discrete_sequence=['blue']
            )
            fig.add_shape(
                type='line',
                x0=ys_test.min(), y0=ys_test.min(),
                x1=ys_test.max(), y1=ys_test.max(),
                line=dict(color='red', dash='dash')
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── S11 Classification accuracy ───────────────────────────
    st.subheader("✅ S11 Classification Accuracy")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Logistic Regression", "96.15%")
    with col2:
        st.metric("Random Forest", "96.15%")
    with col3:
        st.metric("XGBoost", "96.15%")

# ════════════════════════════════════════════════════════════
# PAGE 3 — DATASET EXPLORER
# ════════════════════════════════════════════════════════════
elif page == "🗄️ Dataset Explorer":

    st.title("🗄️ Dataset Explorer")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", "260")
    with col2:
        st.metric("Parameters", "4")
    with col3:
        st.metric("CST Simulations", "400")

    st.markdown("---")

    st.subheader("📋 Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")

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

# ════════════════════════════════════════════════════════════
# PAGE 4 — ANTENNA OPTIMIZER
# ════════════════════════════════════════════════════════════
elif page == "🔧 Antenna Optimizer":

    st.title("🔧 Antenna Inverse Design Optimizer")
    st.markdown("Enter your **desired frequency** — ML will find optimal antenna dimensions instantly.")
    st.markdown("---")

    # ── How it works explanation ──────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\nEnter your target frequency")
    with col2:
        st.info("**Step 2**\nClick Optimize")
    with col3:
        st.info("**Step 3**\nGet optimal dimensions instantly")

    st.markdown("---")

    # ── Input ─────────────────────────────────────────────────
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

            # ── Objective function ────────────────────────────
            def objective(params):
                PL, PW, INL, ML = params
                input_data = np.array([[PL, PW, INL, ML]])
                freq_pred = xgb_freq.predict(input_data)[0]
                s11_pred  = rf_s11.predict(input_data)[0]
                freq_error   = abs(freq_pred - target_freq)
                s11_penalty  = max(0, s11_pred + 10) * 0.1
                return freq_error + s11_penalty

            # ── Run optimization ──────────────────────────────
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

            # ── Get optimal dimensions ────────────────────────
            PL_opt, PW_opt, INL_opt, ML_opt = result.x
            input_opt   = np.array([[PL_opt, PW_opt, INL_opt, ML_opt]])
            freq_pred   = xgb_freq.predict(input_opt)[0]
            s11_pred    = rf_s11.predict(input_opt)[0]
            class_pred  = rf_class.predict(input_opt)[0]
            freq_error  = abs(freq_pred - target_freq)
            freq_error_pct = (freq_error / target_freq) * 100

            # ── Display results ───────────────────────────────
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

            # ── Frequency gauge ───────────────────────────────
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

            # ── S11 gauge ─────────────────────────────────────
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

    # ── Comparison table ──────────────────────────────────────
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