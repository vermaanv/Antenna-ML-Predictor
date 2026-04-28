# ============================================================
# Experiment 2 - Data Preparation Script
# Project: ML-Assisted Microstrip Patch Antenna Design
# Description: Loads CST simulation data, combines parameters
#              with S11 and frequency results, cleans dataset
# Parameters Swept: PL, PW, INL, ML (4 parameters)
# Total Simulations: 400 (Run ID 105 to 504)
# Note: S11 and Frequency files are swapped in CST export
#       so column renaming is applied after merge
# ============================================================

import pandas as pd
import numpy as np

# ── Load Raw CST Simulation Data ──────────────────────────────
# Note: CST exported files have swapped column names
# antenna_s11_data_400.txt  → actually contains Frequency values
# antenna_freq_data_400.txt → actually contains S11 values
s11_data = pd.read_csv('../data/antenna_s11_data_400.txt',
                        sep='\t', comment='#', header=None,
                        names=['run_id', 'S11'])

freq_data = pd.read_csv('../data/antenna_freq_data_400.txt',
                         sep='\t', comment='#', header=None,
                         names=['run_id', 'Frequency'])

# ── Clean Run IDs ─────────────────────────────────────────────
# Convert float run IDs to integers for proper merging
s11_data['run_id']  = s11_data['run_id'].round().astype(int)
freq_data['run_id'] = freq_data['run_id'].round().astype(int)

# ── Filter Only Experiment 2 Runs (105 to 504) ───────────────
# Runs 1-104 belong to Experiment 1 — exclude them
s11_data  = s11_data[(s11_data['run_id'] >= 105) &
                     (s11_data['run_id'] <= 504)]
freq_data = freq_data[(freq_data['run_id'] >= 105) &
                      (freq_data['run_id'] <= 504)]

print(f"S11 rows loaded  : {len(s11_data)}")
print(f"Freq rows loaded : {len(freq_data)}")

# ── Define Swept Parameter Values ────────────────────────────
# PL  = Patch Length (mm)
# PW  = Patch Width (mm)
# INL = Inset Notch Length (mm)
# ML  = Microstrip Line Length / Feed Length (mm)

PL_values  = [26, 27.75, 29.5, 31.25, 33]     # 5 steps
PW_values  = [34, 36.25, 38.5, 40.75, 43]     # 5 steps
INL_values = [6, 7.33333, 8.66667, 10]        # 4 steps
ML_values  = [14, 16, 18, 20]                 # 4 steps
# Total combinations = 5 x 5 x 4 x 4 = 400 simulations

# ── Generate All Parameter Combinations ──────────────────────
params = []
for pl in PL_values:
    for pw in PW_values:
        for inl in INL_values:
            for ml in ML_values:
                params.append([pl, pw, inl, ml])

params_df = pd.DataFrame(params, columns=['PL', 'PW', 'INL', 'ML'])
params_df['run_id'] = range(105, 505)

# ── Merge Parameters with Simulation Results ──────────────────
df = params_df.merge(s11_data, on='run_id')
df = df.merge(freq_data, on='run_id')

# ── Fix Swapped Column Names ──────────────────────────────────
# CST exported files had swapped S11/Frequency labels
# Correcting by renaming columns
df = df.rename(columns={'S11': 'Frequency', 'Frequency': 'S11'})

# ── Clean Bad Data Points ─────────────────────────────────────
# Keep only antennas resonating between 2.0 and 2.8 GHz
# (near target frequency of 2.4 GHz)
# Remove very poor S11 values (> -5 dB)
df = df[df['Frequency'] > 2.0]
df = df[df['Frequency'] < 2.8]
df = df[df['S11'] < -5]

# ── Reset Index & Save ────────────────────────────────────────
df = df.reset_index(drop=True)
df = df.drop(columns=['run_id'])
df.to_csv('../data/antenna_dataset_400.csv', index=False)

# ── Print Summary ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("EXPERIMENT 2 - DATASET PREPARATION COMPLETE")
print("=" * 50)
print(f"Total clean samples : {len(df)}")
print(f"Parameters          : PL, PW, INL, ML")
print(f"Frequency range     : {df['Frequency'].min():.3f} - "
      f"{df['Frequency'].max():.3f} GHz")
print(f"S11 range           : {df['S11'].min():.3f} - "
      f"{df['S11'].max():.3f} dB")
print(f"Output file         : data/antenna_dataset_400.csv")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Statistics:")
print(df.describe())