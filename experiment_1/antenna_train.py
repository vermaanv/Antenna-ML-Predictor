# ============================================================
# Experiment 1 - ML Training Script
# Project: ML-Assisted Microstrip Patch Antenna Design
# Description: Trains 3 ML models to predict S11 and
#              resonant frequency from antenna dimensions
# Input Features : PL, PW, INL (3 parameters)
# Output Targets : S11 (dB), Resonant Frequency (GHz)
# Models Used    : Linear Regression, Random Forest, XGBoost
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ── Load Dataset ──────────────────────────────────────────────
df = pd.read_csv('antenna_dataset.csv')
print(f"Dataset loaded: {len(df)} samples")

# ── Define Features and Targets ───────────────────────────────
X      = df[['PL', 'PW', 'INL']]   # Input: antenna dimensions
y_s11  = df['S11']                  # Target 1: S11 value (dB)
y_freq = df['Frequency']            # Target 2: Resonant frequency (GHz)

# ── Train/Test Split (80% train, 20% test) ────────────────────
X_train, X_test, y_s11_train, y_s11_test = train_test_split(
    X, y_s11, test_size=0.2, random_state=42)
_, _, y_freq_train, y_freq_test = train_test_split(
    X, y_freq, test_size=0.2, random_state=42)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ── Define ML Models ──────────────────────────────────────────
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

# ── Train & Evaluate S11 Prediction Models ────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 1 - S11 PREDICTION RESULTS")
print("=" * 60)

s11_results = {}
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

# ── Train & Evaluate Frequency Prediction Models ──────────────
print("\n" + "=" * 60)
print("EXPERIMENT 1 - FREQUENCY PREDICTION RESULTS")
print("=" * 60)

freq_results = {}
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

# ── Generate Plots ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Experiment 1 - ML Model Performance\n'
             'Antenna Parameter Prediction (3 Parameters, 79 Samples)',
             fontsize=13, fontweight='bold')

# Row 1: S11 prediction plots
for idx, (name, res) in enumerate(s11_results.items()):
    ax = axes[0][idx]
    ax.scatter(y_s11_test, res['pred'],
               color='blue', alpha=0.7, edgecolors='k', linewidths=0.3)
    ax.plot([y_s11_test.min(), y_s11_test.max()],
            [y_s11_test.min(), y_s11_test.max()], 'r--', linewidth=2)
    ax.set_xlabel('Actual S11 (dB)')
    ax.set_ylabel('Predicted S11 (dB)')
    ax.set_title(f'S11 - {name}\nR²={res["r2"]:.3f}, MAE={res["mae"]:.2f}dB')
    ax.grid(True, alpha=0.3)

# Row 2: Frequency prediction plots
for idx, (name, res) in enumerate(freq_results.items()):
    ax = axes[1][idx]
    ax.scatter(y_freq_test, res['pred'],
               color='green', alpha=0.7, edgecolors='k', linewidths=0.3)
    ax.plot([y_freq_test.min(), y_freq_test.max()],
            [y_freq_test.min(), y_freq_test.max()], 'r--', linewidth=2)
    ax.set_xlabel('Actual Frequency (GHz)')
    ax.set_ylabel('Predicted Frequency (GHz)')
    ax.set_title(f'Freq - {name}\nR²={res["r2"]:.3f}, MAE={res["mae"]:.4f}GHz')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as ml_results.png")