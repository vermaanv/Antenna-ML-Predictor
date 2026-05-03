# ============================================================
# Experiment 2 - ML Training Script
# Project: ML-Assisted Microstrip Patch Antenna Design
# Description: Trains 4 ML models to predict S11 and
#              resonant frequency from antenna dimensions
# Input Features : PL, PW, INL, ML (4 parameters)
# Output Targets : S11 (dB), Resonant Frequency (GHz)
# Dataset        : 260 clean samples
# Models Used    : Linear Regression, Random Forest,
#                  XGBoost, ANN (Neural Network)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ── Load Dataset ──────────────────────────────────────────────
df = pd.read_csv('../data/antenna_dataset_400.csv')
print(f"Dataset loaded: {len(df)} samples")

# ── Define Features and Targets ───────────────────────────────
X      = df[['PL', 'PW', 'INL', 'ML']]
y_s11  = df['S11']
y_freq = df['Frequency']

# ── Train/Test Split (80% train, 20% test) ────────────────────
X_train, X_test, y_s11_train, y_s11_test = train_test_split(
    X, y_s11, test_size=0.2, random_state=42)
_, _, y_freq_train, y_freq_test = train_test_split(
    X, y_freq, test_size=0.2, random_state=42)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

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

# ── Define Traditional ML Models ──────────────────────────────
models_s11 = {
    'Linear Regression': LinearRegression(),
    'Random Forest'    : RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost'          : XGBRegressor(n_estimators=100, random_state=42)
}

models_freq = {
    'Linear Regression': LinearRegression(),
    'Random Forest'    : RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost'          : XGBRegressor(n_estimators=100, random_state=42)
}

# ════════════════════════════════════════════════════════════
# S11 PREDICTION
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 2 - S11 PREDICTION RESULTS")
print("=" * 60)

s11_results = {}

# Train traditional models
for name, model in models_s11.items():
    model.fit(X_train, y_s11_train)
    pred = model.predict(X_test)
    mae  = mean_absolute_error(y_s11_test, pred)
    r2   = r2_score(y_s11_test, pred)
    rmse = np.sqrt(mean_squared_error(y_s11_test, pred))
    s11_results[name] = {'pred': pred, 'mae': mae, 'r2': r2, 'rmse': rmse}
    print(f"\n{name}:")
    print(f"  R² Score : {r2:.4f}")
    print(f"  MAE      : {mae:.4f} dB")
    print(f"  RMSE     : {rmse:.4f} dB")

# Train ANN for S11
print("\nTraining ANN for S11 prediction...")
ann_s11 = build_ann()
ann_s11.fit(
    X_train_scaled, y_s11_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)
ann_s11_pred = ann_s11.predict(X_test_scaled).flatten()
ann_s11_mae  = mean_absolute_error(y_s11_test, ann_s11_pred)
ann_s11_r2   = r2_score(y_s11_test, ann_s11_pred)
ann_s11_rmse = np.sqrt(mean_squared_error(y_s11_test, ann_s11_pred))
s11_results['ANN'] = {
    'pred': ann_s11_pred,
    'mae' : ann_s11_mae,
    'r2'  : ann_s11_r2,
    'rmse': ann_s11_rmse
}
print(f"\nANN:")
print(f"  R² Score : {ann_s11_r2:.4f}")
print(f"  MAE      : {ann_s11_mae:.4f} dB")
print(f"  RMSE     : {ann_s11_rmse:.4f} dB")

# ════════════════════════════════════════════════════════════
# FREQUENCY PREDICTION
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 2 - FREQUENCY PREDICTION RESULTS")
print("=" * 60)

freq_results = {}

# Train traditional models
for name, model in models_freq.items():
    model.fit(X_train, y_freq_train)
    pred = model.predict(X_test)
    mae  = mean_absolute_error(y_freq_test, pred)
    r2   = r2_score(y_freq_test, pred)
    rmse = np.sqrt(mean_squared_error(y_freq_test, pred))
    freq_results[name] = {'pred': pred, 'mae': mae, 'r2': r2, 'rmse': rmse}
    print(f"\n{name}:")
    print(f"  R² Score : {r2:.4f}")
    print(f"  MAE      : {mae:.4f} GHz")
    print(f"  RMSE     : {rmse:.4f} GHz")

# Train ANN for Frequency
print("\nTraining ANN for Frequency prediction...")
ann_freq = build_ann()
ann_freq.fit(
    X_train_scaled, y_freq_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)
ann_freq_pred = ann_freq.predict(X_test_scaled).flatten()
ann_freq_mae  = mean_absolute_error(y_freq_test, ann_freq_pred)
ann_freq_r2   = r2_score(y_freq_test, ann_freq_pred)
ann_freq_rmse = np.sqrt(mean_squared_error(y_freq_test, ann_freq_pred))
freq_results['ANN'] = {
    'pred': ann_freq_pred,
    'mae' : ann_freq_mae,
    'r2'  : ann_freq_r2,
    'rmse': ann_freq_rmse
}
print(f"\nANN:")
print(f"  R² Score : {ann_freq_r2:.4f}")
print(f"  MAE      : {ann_freq_mae:.4f} GHz")
print(f"  RMSE     : {ann_freq_rmse:.4f} GHz")

# ── Summary Table ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY - ALL MODELS COMPARISON")
print("=" * 60)
print(f"\n{'Model':<20} {'S11 R²':>10} {'Freq R²':>10}")
print("-" * 42)
for name in s11_results:
    print(f"{name:<20} {s11_results[name]['r2']:>10.4f} {freq_results[name]['r2']:>10.4f}")

# ── Generate Plots ────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Experiment 2 - ML Model Performance (4 Models)\n'
             'Antenna Parameter Prediction (4 Parameters, 260 Samples)',
             fontsize=13, fontweight='bold')

# Row 1: S11 prediction plots
colors = ['blue', 'blue', 'blue', 'red']
for idx, (name, res) in enumerate(s11_results.items()):
    ax = axes[0][idx]
    ax.scatter(y_s11_test, res['pred'],
               color=colors[idx], alpha=0.7,
               edgecolors='k', linewidths=0.3)
    ax.plot([y_s11_test.min(), y_s11_test.max()],
            [y_s11_test.min(), y_s11_test.max()],
            'r--', linewidth=2)
    ax.set_xlabel('Actual S11 (dB)')
    ax.set_ylabel('Predicted S11 (dB)')
    ax.set_title(f'S11 - {name}\nR²={res["r2"]:.3f}, MAE={res["mae"]:.2f}dB')
    ax.grid(True, alpha=0.3)

# Row 2: Frequency prediction plots
colors2 = ['green', 'green', 'green', 'orange']
for idx, (name, res) in enumerate(freq_results.items()):
    ax = axes[1][idx]
    ax.scatter(y_freq_test, res['pred'],
               color=colors2[idx], alpha=0.7,
               edgecolors='k', linewidths=0.3)
    ax.plot([y_freq_test.min(), y_freq_test.max()],
            [y_freq_test.min(), y_freq_test.max()],
            'r--', linewidth=2)
    ax.set_xlabel('Actual Frequency (GHz)')
    ax.set_ylabel('Predicted Frequency (GHz)')
    ax.set_title(f'Freq - {name}\nR²={res["r2"]:.3f}, MAE={res["mae"]:.4f}GHz')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/ml_results_with_ann.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as results/ml_results_with_ann.png")