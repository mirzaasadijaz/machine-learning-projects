# Heart Disease Prediction

A machine learning pipeline that predicts the presence of heart disease from clinical and diagnostic patient data, using the UCI Cleveland Heart Disease dataset and a Support Vector Machine (RBF kernel) classifier.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

## Overview

This project trains a binary classifier that predicts whether a patient has heart disease from 13 clinical features (age, blood pressure, cholesterol, ECG results, etc.). The full workflow — data cleaning, exploratory analysis, preprocessing, model training, and evaluation — lives in `heart_disease_main.ipynb`. The final pipeline reaches **91.8% test accuracy** and is serialized to `heart_model.pkl` for reuse.

## Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease dataset (Cleveland subset)
- **File used:** `dataset/heart+disease/processed.cleveland.data`
- **Samples:** 303 patients, 13 features + 1 target
- **Missing values:** `ca` (4 rows), `thal` (2 rows) — imputed rather than dropped
- **Class balance:** 164 no disease vs. 139 disease

### Features

| Column | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG results (0 = normal, 1 = ST-T wave abnormality, 2 = probable/definite left ventricular hypertrophy) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping) |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) |
| `target` | Diagnosis, binarized from the original 0–4 severity scale (0 = no disease, 1 = disease present) |

## Project Structure

```
.
├── heart_disease_main.ipynb   # Main notebook: EDA, preprocessing, training, evaluation
├── dataset/
│   └── heart+disease/
│       └── processed.cleveland.data
├── heart_model.pkl            # Saved trained pipeline (created after running the notebook)
└── README.md
```

## Methodology

1. **Load & clean** — read the raw `.data` file with named columns, treat `?` as missing, binarize `target` into 0/1.
2. **Explore** — check shape and missing values, plot class balance, rank features by absolute correlation with `target`.
3. **Split** — 80/20 train/test split, stratified on `target`, `random_state=42`.
4. **Preprocess** (`ColumnTransformer`, fit only on the training set):
   - Nominal (`cp`, `restecg`, `slope`, `thal`) → most-frequent imputation + one-hot encoding
   - Numeric (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`) → median imputation + standard scaling
   - Binary (`sex`, `fbs`, `exang`) → most-frequent imputation
5. **Train** — `SVC(kernel='rbf', probability=True)`, wrapped with the preprocessing step in a single `sklearn.pipeline.Pipeline`.
6. **Evaluate** — classification report and confusion matrix on the held-out test set.
7. **Persist** — the fitted pipeline is saved to `heart_model.pkl` with `joblib`.

## Results

Evaluated on the 61-sample held-out test set:

**Test Accuracy: 91.8%**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No Disease | 0.94 | 0.91 | 0.92 | 33 |
| Disease | 0.90 | 0.93 | 0.91 | 28 |
| **Weighted avg** | **0.92** | **0.92** | **0.92** | **61** |

Confusion matrix:

| | Predicted: No Disease | Predicted: Disease |
|---|---|---|
| **Actual: No Disease** | 30 | 3 |
| **Actual: Disease** | 2 | 26 |

## Getting Started

### Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Data

Download the processed Cleveland dataset from the UCI Machine Learning Repository and place it at:

```
dataset/heart+disease/processed.cleveland.data
```

## Usage

Run the full pipeline:

```bash
jupyter notebook heart_disease_main.ipynb
```

Run all cells top to bottom — data loading → EDA → preprocessing → training → evaluation — and the trained pipeline is saved as `heart_model.pkl`.

Reuse the saved model for inference on new patients:

```python
import joblib
import pandas as pd

model = joblib.load("heart_model.pkl")

new_patient = pd.DataFrame([{
    "age": 58, "sex": 1, "cp": 3, "trestbps": 132, "chol": 224,
    "fbs": 0, "restecg": 0, "thalach": 173, "exang": 0,
    "oldpeak": 3.2, "slope": 1, "ca": 2, "thal": 7
}])

prediction = model.predict(new_patient)[0]
probability = model.predict_proba(new_patient)[0][1]
print("Disease" if prediction == 1 else "No Disease", f"({probability:.1%} probability)")
```

Dataset: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X. Licensed under CC BY 4.0.
