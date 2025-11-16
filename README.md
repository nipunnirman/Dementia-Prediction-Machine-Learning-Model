#  Dementia Risk Prediction

> Machine learning model achieving **94.19% ROC-AUC** for dementia prediction using the NACC dataset

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-94.19%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

##  Overview

Binary classification model predicting dementia risk from 195,196 clinical records using XGBoost, Random Forest, and Logistic Regression.

**Key Achievement:** 94.19% ROC-AUC with zero data leakage and minimal overfitting (0.56% gap).

---

##  Results

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|-------|---------|----------|-----------|--------|-----|
| **XGBoost** | **94.19%** | **90.16%** | **83.49%** | **83.05%** | **83.27%** |
| Random Forest | 93.91% | 89.99% | 83.03% | 83.04% | 83.04% |
| Logistic Regression | 93.18% | 89.18% | 84.41% | 77.66% | 80.90% |

---

##  Dataset

- **Source:** NACC (National Alzheimer's Coordinating Center)
- **Samples:** 195,196 total (156,156 train / 39,040 test)
- **Features:** 41 (after preprocessing)
- **Target:** DEMENTED (Binary: 0/1)
- **Class Balance:** 70.5% No Dementia / 29.5% Dementia

---

##  Project Structure

```
├── version_1/                    # ⚠️ First attempt (has preprocessing issues - kept for reference)
│
├── notebooks/                    # ✅ COMPLETE WORKING CODE (Use these!)
│   ├── 01_Data_Loading_and_Exploration.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Preprocessing_and_Scaling.ipynb
│   └── 05_Model_Training.ipynb
│
├── data/                         # Datasets and preprocessed files
├── models/                       # Trained models (.pkl files)
├── outputs/                      # Results, plots, report (PDF)
└── README.md                     # This file
```

### Important Note

- **`version_1/` folder:** Contains my first attempt at building the model. There are issues in the model preprocessing part, but I kept it for reference and learning purposes. **Do not use this version.**
  
- **`notebooks/` folder:** Contains the **complete, working code** split into **5 separate notebook files**. This is the final, production-ready version that achieved 94.19% ROC-AUC. **Use these files!**

---

##  Quick Start

### Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### Run Complete Pipeline

Execute the **5 notebooks in order**:

```bash
jupyter notebook

# Run in sequence:
# 1. 01_Data_Loading_and_Exploration.ipynb  (~30 seconds)
# 2. 02_Data_Cleaning.ipynb                 (~1 minute)
# 3. 03_Feature_Engineering.ipynb           (~1 minute)
# 4. 04_Preprocessing_and_Scaling.ipynb     (~2 minutes)
# 5. 05_Model_Training.ipynb                (~5-10 minutes)
```

**Total Runtime:** ~15 minutes

### Use Pre-trained Model
```python
import pickle

# Load model
with open('models/xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]
```

---

## Methodology

**Complete pipeline across 5 notebooks:**

1. **Notebook 01 - Data Loading:** Load NACC dataset, initial exploration, understand structure
2. **Notebook 02 - Data Cleaning:** Remove 17 ID columns, handle 3.29% missing values
3. **Notebook 03 - Feature Engineering:** Create 18 derived features (age groups, education, social isolation, family history)
4. **Notebook 04 - Preprocessing:** StandardScaler normalization, train-test split (80/20)
5. **Notebook 05 - Model Training:** Train 3 models with 5-fold cross-validation, evaluate performance

---

##  Key Findings

**Top 3 Predictors:**
1. **INDEPEND (72%)** - Functional independence level
2. **months_in_study (3%)** - Study duration
3. **NACCLIVS (3%)** - Living situation

**Quality Assurance:**
- ✅ No data leakage (removed diagnosis columns)
- ✅ Minimal overfitting (0.56% gap)
- ✅ Validated with 5-fold CV (94.31% ± 0.09%)

---

##  Usage

### Training
```python
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train_scaled, y_train)
```

### Evaluation
```python
from sklearn.metrics import roc_auc_score

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {auc:.4f}")
```

---

## ⚠️ Limitations

- INDEPEND feature dominates predictions (72%)
- Trained on NACC dataset only (needs external validation)
- Longitudinal features may limit single-visit predictions

---

##  Generated Files

**Models (in `models/` folder):**
- `xgboost.pkl` - Best model (94.19% ROC-AUC)
- `random_forest.pkl` - 93.91% ROC-AUC
- `logistic_regression.pkl` - 93.18% ROC-AUC

**Results (in `outputs/` folder):**
- `final_model_comparison.csv` - Performance metrics
- `feature_importance.png` - Feature importance visualization
- `roc_curves.png` - ROC curves for all models
- `Dementia_Prediction_Report.pdf` - Complete 14-page report

---

##  Documentation

- **Full Report:** [Dementia_Prediction_Report.pdf](outputs/Dementia_Prediction_Report.pdf)
- **Model Comparison:** [final_model_comparison.csv](outputs/final_model_comparison.csv)

---

##  Workflow

```
Data (195K samples)
    ↓
Notebook 01: Load & Explore
    ↓
Notebook 02: Clean (Remove IDs, Handle Missing)
    ↓
Notebook 03: Engineer Features (18 new features)
    ↓
Notebook 04: Preprocess (Scale, Split 80/20)
    ↓
Notebook 05: Train Models (XGBoost, RF, LR)
    ↓
Results: 94.19% ROC-AUC 
```

---

##  Acknowledgments

- NACC for the dataset
- ModelX Hackathon organizers
- scikit-learn, XGBoost communities

---

##  Contact

**Project:** ModelX Hackathon - Dementia Risk Prediction  
**Status:** Production Ready  
**Date:** November 2024

---



---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**



</div>
