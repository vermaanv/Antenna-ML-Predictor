# ============================================================
# Antenna Inverse Design & Optimization Script
# Project: ML-Assisted Microstrip Patch Antenna Design
# Description: Given desired frequency, finds optimal antenna
#              dimensions using trained ML models
# Method: Scipy optimization with ML surrogate model
# ============================================================

import numpy as np
import pandas as pd
import pickle
import os
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

# ── Load Trained Models ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, '../models/xgb_freq.pkl'), 'rb') as f:
    xgb_freq = pickle.load(f)

with open(os.path.join(BASE_DIR, '../models/rf_s11.pkl'), 'rb') as f:
    rf_s11 = pickle.load(f)

with open(os.path.join(BASE_DIR, '../models/rf_class.pkl'), 'rb') as f:
    rf_class = pickle.load(f)

print("✅ Models loaded successfully")

# ── Define Parameter Bounds ───────────────────────────────────
# Based on our CST parametric sweep ranges
BOUNDS = [
    (26.0, 33.0),   # PL: Patch Length (mm)
    (34.0, 43.0),   # PW: Patch Width (mm)
    (6.0,  10.0),   # INL: Inset Notch Length (mm)
    (14.0, 20.0),   # ML: Feed Line Length (mm)
]

# ── Objective Function ────────────────────────────────────────
def objective(params, target_freq):
    """
    Minimize difference between predicted and target frequency
    Also penalize poor S11 (bad antenna quality)
    """
    PL, PW, INL, ML = params
    input_data = np.array([[PL, PW, INL, ML]])

    # Predict frequency
    freq_pred = xgb_freq.predict(input_data)[0]

    # Predict S11
    s11_pred = rf_s11.predict(input_data)[0]

    # Frequency error (primary objective)
    freq_error = abs(freq_pred - target_freq)

    # S11 penalty (encourage good antenna)
    # If S11 > -10 dB, add penalty
    s11_penalty = max(0, s11_pred + 10) * 0.1

    return freq_error + s11_penalty

# ── Optimization Function ─────────────────────────────────────
def optimize_antenna(target_freq):
    """
    Find optimal antenna dimensions for target frequency
    Uses differential evolution for global optimization
    """
    print(f"\n🔍 Optimizing for target frequency: {target_freq} GHz")
    print("=" * 50)

    # Use differential evolution for global optimization
    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        args=(target_freq,),
        maxiter=1000,
        tol=1e-6,
        seed=42,
        polish=True
    )

    # Extract optimal parameters
    PL_opt, PW_opt, INL_opt, ML_opt = result.x

    # Predict performance with optimal dimensions
    input_opt = np.array([[PL_opt, PW_opt, INL_opt, ML_opt]])
    freq_predicted = xgb_freq.predict(input_opt)[0]
    s11_predicted  = rf_s11.predict(input_opt)[0]
    class_predicted = rf_class.predict(input_opt)[0]

    # Calculate error
    freq_error = abs(freq_predicted - target_freq)
    freq_error_pct = (freq_error / target_freq) * 100

    return {
        'target_freq'  : target_freq,
        'PL'           : round(PL_opt, 2),
        'PW'           : round(PW_opt, 2),
        'INL'          : round(INL_opt, 2),
        'ML'           : round(ML_opt, 2),
        'pred_freq'    : round(freq_predicted, 4),
        'pred_s11'     : round(s11_predicted, 2),
        'antenna_quality': 'GOOD ✅' if class_predicted == 1 else 'POOR ❌',
        'freq_error'   : round(freq_error, 4),
        'freq_error_pct': round(freq_error_pct, 3)
    }

# ── Print Results ─────────────────────────────────────────────
def print_results(res):
    print(f"\n{'='*50}")
    print(f"OPTIMIZATION RESULTS")
    print(f"{'='*50}")
    print(f"Target Frequency    : {res['target_freq']} GHz")
    print(f"{'─'*50}")
    print(f"OPTIMAL DIMENSIONS:")
    print(f"  Patch Length  (PL)  : {res['PL']} mm")
    print(f"  Patch Width   (PW)  : {res['PW']} mm")
    print(f"  Inset Length  (INL) : {res['INL']} mm")
    print(f"  Feed Length   (ML)  : {res['ML']} mm")
    print(f"{'─'*50}")
    print(f"PREDICTED PERFORMANCE:")
    print(f"  Resonant Frequency  : {res['pred_freq']} GHz")
    print(f"  S11 Value           : {res['pred_s11']} dB")
    print(f"  Antenna Quality     : {res['antenna_quality']}")
    print(f"  Frequency Error     : {res['freq_error']} GHz ({res['freq_error_pct']}%)")
    print(f"{'='*50}")

# ── Test Multiple Target Frequencies ─────────────────────────
if __name__ == "__main__":

    # Test different target frequencies
    test_frequencies = [2.4, 2.35, 2.45, 2.5, 2.3]

    all_results = []

    for freq in test_frequencies:
        res = optimize_antenna(freq)
        print_results(res)
        all_results.append(res)

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('../results/optimization_results.csv', index=False)

    print(f"\n✅ Results saved to results/optimization_results.csv")
    print(f"\n{'='*50}")
    print("SUMMARY TABLE")
    print(f"{'='*50}")
    print(results_df[['target_freq', 'pred_freq',
                       'pred_s11', 'antenna_quality',
                       'freq_error_pct']].to_string(index=False))