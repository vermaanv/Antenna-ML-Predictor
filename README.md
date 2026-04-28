# ML-Assisted Microstrip Patch Antenna Design and Performance Prediction

**B.Tech Final Year Project**
**Department of Electronics and Communication Engineering**

---

## Project Overview

This project develops a Machine Learning based surrogate model for microstrip patch antenna design. Instead of running time-consuming CST electromagnetic simulations for every new antenna design, trained ML models predict antenna performance instantly from geometric parameters.

---

## Problem Statement

Traditional antenna design requires full-wave EM simulation in CST Microwave Studio for every parameter change.
- Each simulation takes **3-5 minutes**
- ML model predicts in **milliseconds**
- Achieved **99.94% accuracy** for frequency prediction

---

## Antenna Designed

- **Type:** Rectangular Microstrip Patch Antenna with Inset Feed
- **Target Frequency:** 2.4 GHz (WiFi Band)
- **Substrate:** FR4 (εr = 4.3, h = 1.6 mm)
- **Simulated S11:** -31.77 dB
- **Gain:** 6.15 dBi
- **Tool:** CST Studio Suite (Learning Edition)

---

## Project Structure
Antenna_ML_Predictor/
│
├── experiment_1/
│   ├── antenna_s11_data.txt
│   ├── antenna_freq_data.txt
│   ├── antenna_ml.py
│   ├── antenna_train.py
│   ├── antenna_dataset.csv
│   └── ml_results.png
│
├── experiment_2/
│   ├── data/
│   │   ├── antenna_s11_data_400.txt
│   │   ├── antenna_freq_data_400.txt
│   │   └── antenna_dataset_400.csv
│   ├── scripts/
│   │   ├── antenna_ml_400.py
│   │   ├── antenna_train_400.py
│   │   └── antenna_classification.py
│   └── results/
│       ├── ml_results_400.png
│       └── classification_results.png
│
└── README.md

---

## Methodology

### Step 1 — Antenna Design
- Designed rectangular microstrip patch antenna in CST
- Target frequency: 2.4 GHz on FR4 substrate
- Verified S11 < -10 dB at resonant frequency

### Step 2 — Parametric Sweep
- Varied antenna dimensions in CST parametric sweep
- Collected S11 and frequency for each combination
- Exported results as ASCII text files

### Step 3 — Data Preparation
- Merged parameter combinations with simulation results
- Removed bad data points
- Saved clean dataset as CSV

### Step 4 — ML Training
- Trained 3 regression models for S11 and frequency prediction
- Trained 3 classification models for good/bad antenna detection
- Evaluated using R2, MAE, RMSE metrics

---

## Parameters Used

| Parameter | Description | Range |
|---|---|---|
| PL | Patch Length | 26 - 33 mm |
| PW | Patch Width | 34 - 43 mm |
| INL | Inset Notch Length | 6 - 10 mm |
| ML | Feed Line Length | 14 - 20 mm |

---

## Results

### Experiment 1 (3 Parameters, 74 Samples)

| Target | Model | R2 Score |
|---|---|---|
| S11 | Random Forest | 0.3223 |
| Frequency | Random Forest | 0.9643 |

### Experiment 2 (4 Parameters, 260 Samples)

| Target | Model | R2 Score / Accuracy |
|---|---|---|
| S11 Prediction | Random Forest | 0.3421 |
| Frequency | XGBoost | 0.9994 (99.94%) |
| S11 Classification | All Models | 96.15% |

---

## Key Findings

- XGBoost predicts resonant frequency with 99.94% accuracy
- All 3 models classify antenna quality with 96.15% accuracy
- Adding feed length (ML) as 4th parameter improved frequency
  prediction from 96.43% to 99.94%
- S11 exact prediction is challenging due to its highly
  nonlinear nature

---

## How to Run

### Requirements

pip install pandas numpy scikit-learn xgboost matplotlib seaborn

### Experiment 1

cd experiment_1
python antenna_ml.py
python antenna_train.py

### Experiment 2

cd experiment_2/scripts
python antenna_ml_400.py
python antenna_train_400.py
python antenna_classification.py

---

## Tools and Technologies

| Category | Tool |
|---|---|
| EM Simulation | CST Studio Suite Learning Edition |
| Programming | Python 3.x |
| ML Library | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |

---

## Models Used

| Model | Type | Used For |
|---|---|---|
| Linear Regression | Regression | Baseline prediction |
| Random Forest | Regression/Classification | Best S11 classifier |
| XGBoost | Regression/Classification | Best frequency predictor |

---

## Conclusion

This project successfully demonstrates that ML surrogate models can replace time-consuming EM simulations for antenna performance prediction. XGBoost achieves 99.94% accuracy for resonant frequency prediction, enabling instant antenna design optimization without CST simulation.