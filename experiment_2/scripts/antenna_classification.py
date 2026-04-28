# ============================================================
# Experiment 2 - S11 Classification Script
# Project: ML-Assisted Microstrip Patch Antenna Design
# Description: Classifies antenna as Good or Bad based on
#              S11 value using 3 ML classification models
# Input Features : PL, PW, INL, ML (4 parameters)
# Output Label   : Good Antenna (S11 < -10dB) = 1
#                  Bad Antenna  (S11 >= -10dB) = 0
# Models Used    : Logistic Regression, Random Forest, XGBoost
# Result         : All models achieved 96.15% accuracy
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Load Dataset ──────────────────────────────────────────────
df = pd.read_csv('../data/antenna_dataset_400.csv')
print(f"Dataset loaded: {len(df)} samples")

# ── Define Features ───────────────────────────────────────────
X = df[['PL', 'PW', 'INL', 'ML']]   # Input: antenna dimensions

# ── Create Classification Labels ──────────────────────────────
# Good antenna = S11 < -10 dB  → label = 1
# Bad antenna  = S11 >= -10 dB → label = 0
# -10 dB is the standard threshold for acceptable antenna matching
df['label'] = (df['S11'] < -10).astype(int)
y = df['label']

print(f"\nClass Distribution:")
print(f"  Good antennas (S11 < -10dB)  : {y.sum()}")
print(f"  Bad antennas  (S11 >= -10dB) : {len(y) - y.sum()}")

# ── Train/Test Split (80% train, 20% test) ────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"  Training samples : {len(X_train)}")
print(f"  Testing samples  : {len(X_test)}")

# ── Define Classification Models ──────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost'            : XGBClassifier(n_estimators=100, random_state=42)
}

# ── Train & Evaluate Models ───────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 2 - S11 CLASSIFICATION RESULTS")
print("Good Antenna (S11 < -10dB) vs Bad Antenna (S11 >= -10dB)")
print("=" * 60)

results = {}
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)

    # Predict on test set
    pred = model.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, pred)
    results[name] = {'pred': pred, 'acc': acc}

    print(f"\n{name}:")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Detailed Report:")
    print(classification_report(y_test, pred,
          target_names=['Bad Antenna', 'Good Antenna']))

# ── Generate Confusion Matrix Plots ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Experiment 2 - S11 Classification Results\n'
             'Good Antenna (S11 < -10dB) vs Bad Antenna (S11 >= -10dB)',
             fontsize=13, fontweight='bold')

for idx, (name, res) in enumerate(results.items()):
    ax = axes[idx]

    # Plot confusion matrix
    cm = confusion_matrix(y_test, res['pred'])
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Labels and titles
    ax.set_title(f'{name}\nAccuracy: {res["acc"]*100:.1f}%')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Bad', 'Good'])
    ax.set_yticklabels(['Bad', 'Good'])

    # Add numbers inside confusion matrix cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/classification_results.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as results/classification_results.png")