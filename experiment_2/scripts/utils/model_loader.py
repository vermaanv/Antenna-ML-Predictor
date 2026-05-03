# ============================================================
# Model Loader Utility
# Shared function to load all trained models and dataset
# Used by all pages
# ============================================================

import os
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_models():
    models_dir = os.path.join(BASE_DIR, '../models')
    with open(os.path.join(models_dir, 'xgb_freq.pkl'), 'rb') as f:
        xgb_freq = pickle.load(f)
    with open(os.path.join(models_dir, 'rf_s11.pkl'), 'rb') as f:
        rf_s11 = pickle.load(f)
    with open(os.path.join(models_dir, 'rf_class.pkl'), 'rb') as f:
        rf_class = pickle.load(f)
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    ann_s11  = load_model(os.path.join(models_dir, 'ann_s11.keras'))
    ann_freq = load_model(os.path.join(models_dir, 'ann_freq.keras'))
    return xgb_freq, rf_s11, rf_class, scaler, ann_s11, ann_freq

@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, '../data/antenna_dataset_400.csv')
    import pandas as pd
    return pd.read_csv(data_path)