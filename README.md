# 💳 Credit Card Fraud Detection — Sampling Techniques Study

A comparative analysis of resampling strategies for handling class imbalance in credit card fraud detection, evaluated across multiple classifiers using Accuracy and F1 Score.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Class Distribution](#class-distribution)
- [Preprocessing](#preprocessing)
- [Sampling Techniques](#sampling-techniques)
- [Models](#models)
- [Results](#results)
- [Visualizations](#visualizations)

---

## 🔍 Overview

Credit card fraud datasets are highly imbalanced — fraudulent transactions make up a tiny fraction of all records. This notebook explores how three different resampling techniques affect model performance, benchmarked across five classifiers.

---

## 📂 Dataset

**File:** `Creditcard_data.csv`  
**Target Column:** `Class` (0 = Legitimate ✅, 1 = Fraudulent 🚨)

---

### 📊 Class Distribution Plot

<img width="463" height="393" alt="image" src="https://github.com/user-attachments/assets/36dfbf09-81c6-446a-894c-5ba6de749085" />
---

## ⚙️ Preprocessing

- **Train/Test Split:** 70% train, 30% test (`stratify=y`, `random_state=7`)
- **Scaling:** MinMaxScaler applied to features

---

## 🔄 Sampling Techniques

| Technique | Method | Type |
|---|---|---|
| `Under_NearMiss` | NearMiss | 📉 Under-sampling |
| `Over_SMOTE` | SMOTE (`k_neighbors=3`) | 📈 Over-sampling |
| `Hybrid_SMOTETomek` | SMOTETomek | 🔀 Hybrid |

### 📊 Class Distribution After Resampling

<!-- INSERT: 2x2 subplot grid showing class distribution after each sampling method -->

<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/655ee484-5cd9-4014-b997-97523ac0c3f7" />

---

## 🤖 Models

| Model | Key Hyperparameters |
|---|---|
| RidgeClassifier | `alpha=1.0` |
| ExtraTreeClassifier | `max_depth=10` |
| GradientBoostingClassifier | `n_estimators=150`, `learning_rate=0.05` |
| AdaBoostClassifier | `n_estimators=120` |
| GaussianNB | — |

---

## 📈 Results

### 🎯 Accuracy Table (%)

| Model | Under_NearMiss | Over_SMOTE | Hybrid_SMOTETomek |
|---|---|---|---|
| Ridge | 21.55 | 80.6 | 82.33 |
| ExtraTree | 16.81 | 87.93 | 84.05 |
| GradientBoost | 34.05 | 98.28 | 98.28 |
| AdaBoost | 38.79 | 98.71 | 98.28 |
| NaiveBayes | 6.47 | 74.57 | 72.84 |


```
...
```

### 🏅 F1 Score Table (%)


| Model | Under_NearMiss | Over_SMOTE | Hybrid_SMOTETomek |
|---|---|---|---|
| Ridge | 3.19 | 4.26 | 4.65 |
| ExtraTree | 3.02 | 6.67 | 0.0 |
| GradientBoost | 1.29 | 0.0 | 0.0 |
| AdaBoost | 1.39 | 0.0 | 0.0 |
| NaiveBayes | 1.81 | 3.28 | 3.08 |

```
...
```

Both tables are also exported as:
- 📄 `modified_accuracy_table.csv`
- 📄 `modified_f1score_table.csv`

---

## 🎨 Visualizations

### 🌡️ F1 Score Heatmap (Models vs Sampling Techniques)

<!-- INSERT: Seaborn heatmap of F1 scores here -->

<img width="631" height="547" alt="image" src="https://github.com/user-attachments/assets/936e5293-d426-4bc6-808e-71af5ff9a5f6" />

---

## 📦 Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | 🗃️ Data manipulation |
| `scikit-learn` | 🤖 Models, preprocessing, metrics |
| `imbalanced-learn` | ⚖️ SMOTE, NearMiss, SMOTETomek |
| `matplotlib`, `seaborn` | 🎨 Visualization |

---



## 💡 Key Findings

### 🏆 Best Performing Models
- **GradientBoost** and **AdaBoost** achieved the highest accuracy of **98.28%** and **98.71%** respectively under both `Over_SMOTE` and `Hybrid_SMOTETomek` — making them the top performers overall.
- **NaiveBayes** struggled the most, with accuracy as low as **6.47%** under `Under_NearMiss`, indicating it is poorly suited for this imbalanced dataset regardless of sampling strategy.

### 📉 Under-Sampling Hurts Performance
- `Under_NearMiss` consistently produced the **worst results** across all models, with accuracies ranging from just **6.47% to 38.79%**. Aggressively removing majority class samples leads to significant loss of useful information.

### 📈 Over-Sampling & Hybrid are Superior
- `Over_SMOTE` and `Hybrid_SMOTETomek` delivered **dramatically better accuracy**, with most models crossing **80–98%**, showing that generating synthetic minority samples is far more effective than discarding majority samples.

### 🌡️ F1 Score Tells a Different Story
- Despite high accuracy, **F1 scores were surprisingly low** for most model-sampling combinations, with many registering **0.0** — particularly GradientBoost and AdaBoost under Over_SMOTE and Hybrid_SMOTETomek. This suggests these models may be **overpredicting the majority class**, inflating accuracy while missing actual fraud cases.
- **ExtraTree + Over_SMOTE** achieved the highest F1 score of **6.7**, making it the best balance between precision and recall.
- **Ridge Classifier** showed the most consistent (though modest) F1 scores across all three sampling methods, suggesting better generalization to the minority class.

