# Credit Card Fraud Detection — Sampling Techniques Study

A comparative analysis of resampling strategies for handling class imbalance in credit card fraud detection, evaluated across multiple classifiers using Accuracy and F1 Score.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Class Distribution](#class-distribution)
- [Preprocessing](#preprocessing)
- [Sampling Techniques](#sampling-techniques)
- [Models](#models)
- [Results](#results)
- [Visualizations](#visualizations)

---

## Overview

Credit card fraud datasets are highly imbalanced — fraudulent transactions make up a tiny fraction of all records. This notebook explores how three different resampling techniques affect model performance, benchmarked across five classifiers.

---

## Dataset

**File:** `Creditcard_data.csv`  
**Target Column:** `Class` (0 = Legitimate, 1 = Fraudulent)

### Preview — `df.head()`

<!-- INSERT: Table / screenshot of df.head() output here -->

```
...
```

**Shape:** <!-- INSERT: df.shape output here, e.g. (10000, 31) -->

---

## Class Distribution

**Class Value Counts (normalized):**

<!-- INSERT: Output of df["Class"].value_counts(normalize=True) here -->

```
...
```

### Class Distribution Plot

<!-- INSERT: countplot of Class distribution here -->

![Class Distribution](images/class_distribution.png)

---

## Preprocessing

- **Train/Test Split:** 70% train, 30% test (`stratify=y`, `random_state=7`)
- **Scaling:** MinMaxScaler applied to features

---

## Sampling Techniques

| Technique | Method | Type |
|---|---|---|
| `Under_NearMiss` | NearMiss | Under-sampling |
| `Over_SMOTE` | SMOTE (`k_neighbors=3`) | Over-sampling |
| `Hybrid_SMOTETomek` | SMOTETomek | Hybrid |

### Class Distribution After Resampling

<!-- INSERT: 2x2 subplot grid showing class distribution after each sampling method -->

![Resampled Distributions](images/resampled_distributions.png)

---

## Models

| Model | Key Hyperparameters |
|---|---|
| RidgeClassifier | `alpha=1.0` |
| ExtraTreeClassifier | `max_depth=10` |
| GradientBoostingClassifier | `n_estimators=150`, `learning_rate=0.05` |
| AdaBoostClassifier | `n_estimators=120` |
| GaussianNB | — |

---

## Results

### Accuracy Table (%)

<!-- INSERT: results_accuracy DataFrame table here -->

```
...
```

### F1 Score Table (%)

<!-- INSERT: results_f1 DataFrame table here -->

```
...
```

Both tables are also exported as:
- `modified_accuracy_table.csv`
- `modified_f1score_table.csv`

---

## Visualizations

### F1 Score Heatmap (Models vs Sampling Techniques)

<!-- INSERT: Seaborn heatmap of F1 scores here -->

![F1 Score Heatmap](images/f1_heatmap.png)

---

## Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Models, preprocessing, metrics |
| `imbalanced-learn` | SMOTE, NearMiss, SMOTETomek |
| `matplotlib`, `seaborn` | Visualization |

---

## How to Run

```bash
jupyter notebook Sampling.ipynb
```

---

## Key Findings

<!-- INSERT: Summary of best-performing model + sampling technique combination based on results -->

```
...
```
