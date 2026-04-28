# ============================================================
# Experiment 1 - Data Preparation Script
# Project: ML-Assisted Microstrip Patch Antenna Design
# Description: Loads CST simulation data, combines parameters
#              with S11 and frequency results, cleans dataset
# Parameters Swept: PL, PW, INL (3 parameters)
# Total Simulations: 100
# ============================================================

import pandas as pd
import numpy as np

# ── Load Raw CST Simulation Data ──────────────────────────────
s11_data = pd.read_csv('antenna_s11_data.txt',
                        sep='\t',
                        comment='#',
                        header=None,
                        names=['run_id', 'S11'])

freq_data = pd.read_csv('antenna_freq_data.txt',
                         sep='\t',
                         comment='#',
                         header=None,
                         names=['run_id', 'Frequency'])

# ── Define Swept Parameter Values ────────────────────────────
# PL  = Patch Length (mm)
# PW  = Patch Width (mm)
# INL = Inset Notch Length (mm)

PL_values  = [26, 27.75, 29.5, 31.25, 33]      # 5 steps
PW_values  = [34, 36.25, 38.5, 40.75, 43]      # 5 steps
INL_values = [6, 7.33333, 8.66667, 10]         # 4 steps
# Total combinations = 5 x 5 x 4 = 100 simulations

# ── Generate All Parameter Combinations ──────────────────────
params = []
for pl in PL_values:
    for pw in PW_values:
        for inl in INL_values:
            params.append([pl, pw, inl])

params_df = pd.DataFrame(params, columns=['PL', 'PW', 'INL'])
params_df['run_id'] = range(5, 5 + len(params_df))

# ── Merge Parameters with Simulation Results ──────────────────
df = params_df.merge(s11_data, on='run_id')
df = df.merge(freq_data, on='run_id')

# ── Clean Bad Data Points ─────────────────────────────────────
# Frequency = 4.0 means antenna did not resonate in range
# S11 > -8  means very poor impedance matching
# S11 < -40 means extreme outlier
df = df[df['Frequency'] < 3.5]
df = df[df['S11'] < -8]
df = df[df['S11'] > -40]

# ── Reset Index & Save ────────────────────────────────────────
df = df.reset_index(drop=True)
df.to_csv('antenna_dataset.csv', index=False)

# ── Print Summary ─────────────────────────────────────────────
print("=" * 50)
print("EXPERIMENT 1 - DATASET PREPARATION COMPLETE")
print("=" * 50)
print(f"Total clean samples : {len(df)}")
print(f"Parameters          : PL, PW, INL")
print(f"Output file         : antenna_dataset.csv")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Statistics:")
print(df.describe())