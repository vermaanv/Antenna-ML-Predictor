# ============================================================
# Page 2 — Model Performance
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_loader import load_models, load_data

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
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

# ── Load Models & Data ────────────────────────────────────────
xgb_freq, rf_s11, rf_class, scaler, ann_s11, ann_freq = load_models()
df = load_data()

# ── Page Content ──────────────────────────────────────────────
st.title("📊 Model Performance")
st.markdown("---")

# ── Experiment comparison table ───────────────────────────────
st.subheader("🔬 Experiment Comparison")
comp_data = {
    'Experiment': ['Experiment 1', 'Experiment 2', 'Experiment 2 + ANN'],
    'Parameters': ['PL, PW, INL (3)', 'PL, PW, INL, ML (4)', 'PL, PW, INL, ML (4)'],
    'Samples': [74, 260, 260],
    'Best Freq R²': [0.9643, 0.9994, 0.9994],
    'Best S11 R²': [0.3223, 0.3421, 0.7010],
    'S11 Classification': ['N/A', '96.15%', '96.15%']
}
comp_df = pd.DataFrame(comp_data)
st.dataframe(comp_df, use_container_width=True)
st.markdown("---")

# ── Prepare data ──────────────────────────────────────────────
X      = df[['PL', 'PW', 'INL', 'ML']]
y_freq = df['Frequency']
y_s11  = df['S11']

X_train, X_test, yf_train, yf_test = train_test_split(
    X, y_freq, test_size=0.2, random_state=42)
_, _, ys_train, ys_test = train_test_split(
    X, y_s11, test_size=0.2, random_state=42)

X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train traditional models ──────────────────────────────────
with st.spinner('Loading model results...'):
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
        pred = model.predict(X_test)
        freq_preds[name] = {
            'pred': pred,
            'r2'  : r2_score(yf_test, pred),
            'mae' : mean_absolute_error(yf_test, pred)
        }

    ann_freq_pred_p2 = ann_freq.predict(X_test_scaled, verbose=0).flatten()
    freq_preds['ANN'] = {
        'pred': ann_freq_pred_p2,
        'r2'  : r2_score(yf_test, ann_freq_pred_p2),
        'mae' : mean_absolute_error(yf_test, ann_freq_pred_p2)
    }

    s11_preds = {}
    for name, model in models_s11.items():
        model.fit(X_train, ys_train)
        pred = model.predict(X_test)
        s11_preds[name] = {
            'pred': pred,
            'r2'  : r2_score(ys_test, pred),
            'mae' : mean_absolute_error(ys_test, pred)
        }

    ann_s11_pred_p2 = ann_s11.predict(X_test_scaled, verbose=0).flatten()
    s11_preds['ANN'] = {
        'pred': ann_s11_pred_p2,
        'r2'  : r2_score(ys_test, ann_s11_pred_p2),
        'mae' : mean_absolute_error(ys_test, ann_s11_pred_p2)
    }

# ── R² bar charts ─────────────────────────────────────────────
st.subheader("📈 R² Score Comparison")
col1, col2 = st.columns(2)

with col1:
    r2_freq_vals = [v['r2'] for v in freq_preds.values()]
    fig = px.bar(
        x=list(freq_preds.keys()),
        y=r2_freq_vals,
        title='Frequency Prediction R² — All 4 Models',
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
        title='S11 Prediction R² — All 4 Models',
        labels={'x': 'Model', 'y': 'R² Score'},
        color=r2_s11_vals,
        color_continuous_scale='Blues'
    )
    fig2.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Frequency Actual vs Predicted ─────────────────────────────
st.subheader("🎯 Frequency Prediction — Actual vs Predicted")
col1, col2, col3, col4 = st.columns(4)

for idx, (name, res) in enumerate(freq_preds.items()):
    with [col1, col2, col3, col4][idx]:
        color = 'orange' if name == 'ANN' else 'green'
        fig = px.scatter(
            x=yf_test, y=res['pred'],
            labels={'x': 'Actual (GHz)', 'y': 'Predicted (GHz)'},
            title=f'{name}\nR²={res["r2"]:.4f}',
            color_discrete_sequence=[color]
        )
        fig.add_shape(
            type='line',
            x0=yf_test.min(), y0=yf_test.min(),
            x1=yf_test.max(), y1=yf_test.max(),
            line=dict(color='red', dash='dash')
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── S11 Actual vs Predicted ───────────────────────────────────
st.subheader("📉 S11 Prediction — Actual vs Predicted")
col1, col2, col3, col4 = st.columns(4)

for idx, (name, res) in enumerate(s11_preds.items()):
    with [col1, col2, col3, col4][idx]:
        color = 'red' if name == 'ANN' else 'blue'
        fig = px.scatter(
            x=ys_test, y=res['pred'],
            labels={'x': 'Actual S11 (dB)', 'y': 'Predicted S11 (dB)'},
            title=f'{name}\nR²={res["r2"]:.4f}',
            color_discrete_sequence=[color]
        )
        fig.add_shape(
            type='line',
            x0=ys_test.min(), y0=ys_test.min(),
            x1=ys_test.max(), y1=ys_test.max(),
            line=dict(color='red', dash='dash')
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── S11 Classification accuracy ───────────────────────────────
st.subheader("✅ S11 Classification Accuracy")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Logistic Regression", "96.15%")
with col2:
    st.metric("Random Forest", "96.15%")
with col3:
    st.metric("XGBoost", "96.15%")