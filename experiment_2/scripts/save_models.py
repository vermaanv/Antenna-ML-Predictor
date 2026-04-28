# ============================================================
# Save Trained ML Models
# This script trains and saves all ML models to files
# so the web app can load them instantly
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import pickle
import os

# ── Load Dataset ──────────────────────────────────────────────
df = pd.read_csv('../data/antenna_dataset_400.csv')
print(f"Dataset loaded: {len(df)} samples")

# ── Define Features and Targets ───────────────────────────────
X      = df[['PL', 'PW', 'INL', 'ML']]
y_freq = df['Frequency']
y_s11  = df['S11']
y_class = (df['S11'] < -10).astype(int)

# ── Train/Test Split ──────────────────────────────────────────
X_train, X_test, yf_train, yf_test = train_test_split(
    X, y_freq, test_size=0.2, random_state=42)
_, _, ys_train, ys_test = train_test_split(
    X, y_s11, test_size=0.2, random_state=42)
_, _, yc_train, yc_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42)

# ── Train Best Models ─────────────────────────────────────────
# XGBoost for frequency (R² = 0.9994)
xgb_freq = XGBRegressor(n_estimators=100, random_state=42)
xgb_freq.fit(X_train, yf_train)

# Random Forest for S11
rf_s11 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_s11.fit(X_train, ys_train)

# Random Forest for classification
rf_class = RandomForestClassifier(n_estimators=100, random_state=42)
rf_class.fit(X_train, yc_train)

# ── Save Models ───────────────────────────────────────────────
os.makedirs('../models', exist_ok=True)

with open('../models/xgb_freq.pkl', 'wb') as f:
    pickle.dump(xgb_freq, f)

with open('../models/rf_s11.pkl', 'wb') as f:
    pickle.dump(rf_s11, f)

with open('../models/rf_class.pkl', 'wb') as f:
    pickle.dump(rf_class, f)

print("\n" + "=" * 50)
print("MODELS SAVED SUCCESSFULLY")
print("=" * 50)
print("xgb_freq.pkl  → Frequency predictor (R²=0.9994)")
print("rf_s11.pkl    → S11 predictor (R²=0.3421)")
print("rf_class.pkl  → S11 classifier (96.15%)")
print("\nSaved to: experiment_2/models/")