# ============================================================
# Save Trained ML Models
# This script trains and saves all ML models to files
# so the web app can load them instantly
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except:
    from tf_keras.models import Sequential
    from tf_keras.layers import Dense, Dropout
    from tf_keras.callbacks import EarlyStopping

import pickle

# ── Load Dataset ──────────────────────────────────────────────
df = pd.read_csv('../data/antenna_dataset_400.csv')
print(f"Dataset loaded: {len(df)} samples")

# ── Define Features and Targets ───────────────────────────────
X       = df[['PL', 'PW', 'INL', 'ML']]
y_freq  = df['Frequency']
y_s11   = df['S11']
y_class = (df['S11'] < -10).astype(int)

# ── Train/Test Split ──────────────────────────────────────────
X_train, X_test, yf_train, yf_test = train_test_split(
    X, y_freq, test_size=0.2, random_state=42)
_, _, ys_train, ys_test = train_test_split(
    X, y_s11, test_size=0.2, random_state=42)
_, _, yc_train, yc_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42)

# ── Scale data for ANN ────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Build ANN Model ───────────────────────────────────────────
def build_ann(input_dim=4):
    model = Sequential([
        Dense(64,  activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64,  activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# ── Train Traditional Models ──────────────────────────────────
print("\nTraining traditional ML models...")

# XGBoost for frequency (R² = 0.9994)
xgb_freq = XGBRegressor(n_estimators=100, random_state=42)
xgb_freq.fit(X_train, yf_train)

# Random Forest for S11
rf_s11 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_s11.fit(X_train, ys_train)

# Random Forest for classification
rf_class = RandomForestClassifier(n_estimators=100, random_state=42)
rf_class.fit(X_train, yc_train)

# ── Train ANN Models ──────────────────────────────────────────
print("Training ANN for S11 prediction...")
ann_s11 = build_ann()
ann_s11.fit(
    X_train_scaled, ys_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

print("Training ANN for Frequency prediction...")
ann_freq = build_ann()
ann_freq.fit(
    X_train_scaled, yf_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# ── Save Traditional Models ───────────────────────────────────
os.makedirs('../models', exist_ok=True)

with open('../models/xgb_freq.pkl', 'wb') as f:
    pickle.dump(xgb_freq, f)

with open('../models/rf_s11.pkl', 'wb') as f:
    pickle.dump(rf_s11, f)

with open('../models/rf_class.pkl', 'wb') as f:
    pickle.dump(rf_class, f)

# ── Save Scaler ───────────────────────────────────────────────
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ── Save ANN Models ───────────────────────────────────────────
ann_s11.save('../models/ann_s11.keras')
ann_freq.save('../models/ann_freq.keras')

print("\n" + "=" * 50)
print("ALL MODELS SAVED SUCCESSFULLY")
print("=" * 50)
print("xgb_freq.pkl   → XGBoost Frequency predictor")
print("rf_s11.pkl     → Random Forest S11 predictor")
print("rf_class.pkl   → Random Forest S11 classifier")
print("ann_s11.keras  → ANN S11 predictor")
print("ann_freq.keras → ANN Frequency predictor")
print("scaler.pkl     → StandardScaler for ANN")
print("\nSaved to: experiment_2/models/")
print("\nRun antenna_train_400.py to see R² scores")