# ML-Assisted Microstrip Patch Antenna Design and Performance Prediction

**B.Tech Final Year Project**
**Department of Electronics and Communication Engineering**

---

## 🌐 Live Web Application

**Try it here:** https://antenna-ml-predictor.streamlit.app

The deployed web application includes:
- 🎯 Antenna Performance Predictor
- 📊 Model Performance Dashboard
- 🗄️ Dataset Explorer
- 🔧 Inverse Design Optimizer

---

## Project Overview

This project develops an end-to-end Machine Learning based surrogate model for microstrip patch antenna design. The system works in two directions:

**Forward Design:**
Antenna Dimensions → ML Model → Predicted Performance
(PL, PW, INL, ML)              (Frequency, S11)

**Inverse Design (New):**
Desired Frequency → ML Optimizer → Optimal Dimensions
(e.g. 2.4 GHz)                    (PL, PW, INL, ML)

Instead of running time-consuming CST electromagnetic simulations for every new antenna design, trained ML models predict antenna performance instantly from geometric parameters.

---

## Problem Statement

Traditional antenna design requires full-wave EM simulation in CST Microwave Studio for every parameter change.

| Method | Time Required |
|---|---|
| CST Simulation | 3-5 minutes per design |
| ML Prediction | Milliseconds |
| ML Optimization | ~15-20 seconds |

---

## Antenna Designed in CST

- **Type:** Rectangular Microstrip Patch Antenna with Inset Feed
- **Target Frequency:** 2.4 GHz (WiFi Band)
- **Substrate:** FR4 (εr = 4.3, h = 1.6 mm)
- **Achieved S11:** -31.77 dB (excellent — only 0.07% power reflected)
- **Gain:** 6.15 dBi
- **Resonant Frequency:** 2.401 GHz
- **Tool:** CST Studio Suite (Learning Edition)

### Antenna Layers:
Top    → Copper patch + inset feed line
Middle → FR4 substrate (1.6mm)
Bottom → Ground plane (copper)

---

## Complete Methodology

### Step 1 — Antenna Design in CST
- Designed rectangular microstrip patch antenna
- Target: 2.4 GHz on FR4 substrate (εr=4.3, h=1.6mm)
- Used inset feed for better impedance matching
- Verified S11 = -31.77 dB at 2.401 GHz
- Verified gain = 6.15 dBi
- Checked 2D and 3D radiation patterns

### Step 2 — Parametric Sweep (400 Simulations)
- Varied 4 antenna dimensions in CST parametric sweep
- Each combination runs full EM simulation
- Exported S11 minimum and resonant frequency
- Total: 400 simulations over 2 nights

### Step 3 — Data Preparation
- Loaded raw CST simulation results
- Matched parameter combinations with simulation results
- Filtered bad data (no resonance, poor S11)
- Final clean dataset: 260 samples

### Step 4 — ML Model Training
- Trained 3 regression models for frequency prediction
- Trained 3 regression models for S11 prediction
- Trained 3 classification models for good/bad antenna
- Evaluated using R², MAE, RMSE metrics

### Step 5 — Inverse Design Optimizer
- Used Differential Evolution optimization algorithm
- Combined with trained XGBoost surrogate model
- Given target frequency → finds optimal dimensions
- Frequency error less than 1% for most targets

### Step 6 — Web Application Deployment
- Built interactive web app using Streamlit
- Deployed live on Streamlit Cloud
- 4 pages: Predictor, Performance, Dataset, Optimizer

---

## Parameters Swept in CST

| Parameter | Description | Range | Steps |
|---|---|---|---|
| PL | Patch Length | 26 - 33 mm | 5 |
| PW | Patch Width | 34 - 43 mm | 5 |
| INL | Inset Notch Length | 6 - 10 mm | 4 |
| ML | Feed Line Length | 14 - 20 mm | 4 |

**Total combinations = 5×5×4×4 = 400 simulations**

---

## Two Experiments

### Experiment 1 — Initial Study
- **Parameters:** PL, PW, INL (3 parameters)
- **Simulations:** 100
- **Clean samples:** 74
- **Goal:** Baseline ML performance

### Experiment 2 — Improved Study
- **Parameters:** PL, PW, INL, ML (4 parameters)
- **Simulations:** 400
- **Clean samples:** 260
- **Goal:** Improved ML performance with more data and parameters

---

## Results

### Experiment 1 (3 Parameters, 74 Samples)

| Target | Best Model | R² Score |
|---|---|---|
| Frequency | Random Forest | 0.9643 (96.43%) |
| S11 | Random Forest | 0.3223 |

### Experiment 2 (4 Parameters, 260 Samples)

| Target | Best Model | R² Score / Accuracy |
|---|---|---|
| Frequency | XGBoost | **0.9994 (99.94%)** ✅ |
| S11 Prediction | Random Forest | 0.3421 |
| S11 Classification | All Models | **96.15%** ✅ |

### Inverse Design Optimizer Results

| Target Freq | Predicted Freq | Error | S11 | Quality |
|---|---|---|---|---|
| 2.40 GHz | 2.3999 GHz | **0.004%** | -18.19 dB | GOOD ✅ |
| 2.35 GHz | 2.3520 GHz | 0.086% | -10.01 dB | POOR ❌ |
| 2.45 GHz | 2.4294 GHz | 0.841% | -24.27 dB | GOOD ✅ |
| 2.50 GHz | 2.5056 GHz | 0.225% | -18.41 dB | GOOD ✅ |
| 2.30 GHz | 2.2907 GHz | 0.404% | -10.58 dB | GOOD ✅ |

---

## Key Findings

- XGBoost predicts resonant frequency with **99.94% accuracy**
- All 3 models classify antenna quality with **96.15% accuracy**
- Adding feed length (ML) as 4th parameter improved frequency prediction from 96.43% to 99.94%
- S11 exact prediction is challenging due to highly nonlinear nature — known limitation in antenna ML research
- Inverse design optimizer achieves less than 1% frequency error for most target frequencies
- Differential Evolution finds optimal dimensions in ~15-20 seconds vs hours of manual CST tuning

---

## Project Structure
Antenna_ML_Predictor/
│
├── experiment_1/
│   ├── antenna_ml.py              ← Data preparation (3 params)
│   ├── antenna_train.py           ← ML training (3 params)
│   ├── antenna_dataset.csv        ← Clean dataset (74 samples)
│   └── ml_results.png             ← Result plots
│
├── experiment_2/
│   ├── data/
│   │   └── antenna_dataset_400.csv    ← Clean dataset (260 samples)
│   │
│   ├── models/
│   │   ├── xgb_freq.pkl           ← XGBoost frequency model (R²=0.9994)
│   │   ├── rf_s11.pkl             ← Random Forest S11 model
│   │   └── rf_class.pkl           ← Random Forest classifier (96.15%)
│   │
│   ├── scripts/
│   │   ├── antenna_ml_400.py      ← Data preparation (4 params)
│   │   ├── antenna_train_400.py   ← ML training (4 params)
│   │   ├── antenna_classification.py ← S11 classification
│   │   ├── antenna_optimizer.py   ← Inverse design optimizer
│   │   ├── save_models.py         ← Save trained models
│   │   └── antenna_app.py         ← Streamlit web application
│   │
│   └── results/
│       ├── ml_results_400.png           ← ML prediction plots
│       ├── classification_results.png   ← Classification plots
│       └── optimization_results.csv     ← Optimizer results
│
├── requirements.txt               ← Python dependencies
└── README.md                      ← This file

---

## How to Run

### Requirements
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit plotly scipy

### Experiment 1
cd experiment_1
python antenna_ml.py
python antenna_train.py

### Experiment 2
cd experiment_2/scripts
python antenna_ml_400.py
python antenna_train_400.py
python antenna_classification.py

### Inverse Design Optimizer
cd experiment_2/scripts
python antenna_optimizer.py

### Web Application (Local)
cd experiment_2/scripts
python -m streamlit run antenna_app.py

### Web Application (Live)
https://antenna-ml-predictor.streamlit.app

---

## Tools and Technologies

| Category | Tool | Purpose |
|---|---|---|
| EM Simulation | CST Studio Suite Learning Edition | Antenna design and parametric sweep |
| Programming | Python 3.x | ML pipeline and web app |
| ML Library | Scikit-learn | Linear Regression, Random Forest |
| ML Library | XGBoost | Best frequency predictor |
| Optimization | Scipy | Differential Evolution optimizer |
| Data Processing | Pandas, NumPy | Dataset preparation |
| Visualization | Matplotlib, Plotly | Result plots and charts |
| Web Framework | Streamlit | Interactive web application |
| Deployment | Streamlit Cloud | Live web deployment |
| Version Control | Git, GitHub | Code management |

---

## ML Models Used

| Model | Type | Used For | Best Result |
|---|---|---|---|
| Linear Regression | Regression | Baseline | Freq R²=0.9934 |
| Random Forest | Regression + Classification | S11 classifier | Class=96.15% |
| XGBoost | Regression | Frequency prediction | Freq R²=0.9994 |
| Differential Evolution | Optimization | Inverse design | Error < 1% |

---

## Why S11 Prediction is Challenging

S11 (return loss) is highly nonlinear — small parameter changes cause large S11 jumps. This is a known challenge in antenna ML research. Therefore we approached S11 in two ways:

1. **Regression** — predict exact S11 value (R²=0.34, limited by nonlinearity)
2. **Classification** — predict good/bad antenna (96.15% accuracy, practical and useful)

The -10 dB threshold is the industry standard — at this point 90% of input power is radiated.

---

## Conclusion

This project successfully demonstrates an end-to-end ML-assisted antenna design system:

- **Forward prediction:** XGBoost predicts resonant frequency with 99.94% accuracy
- **Quality assessment:** All models classify antenna quality with 96.15% accuracy
- **Inverse design:** Optimizer finds optimal dimensions for any target frequency with less than 1% error
- **Deployed application:** Live web app accessible from any browser

The system eliminates the need for repetitive CST simulations — reducing design time from hours to seconds.