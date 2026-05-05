# Supervised Learning Home Project

End-to-end supervised learning project covering regression and classification,
built as part of the ECOM School AI & Data Science course.
Each part follows the full ML workflow: data preparation, exploratory analysis,
hyperparameter tuning, model evaluation, and deployment.

---

## Project Structure

```
ecom--home-project-II/
├── data/
│   ├── Covid19_With_GDP_Values.csv
│   └── customer_churn_dataset.csv
├── docs/
│   └── Supervised Learning - Home Project.pdf
├── models/
│   ├── ridge_model.joblib
│   ├── scaler_gdp.joblib
│   ├── random_forest_model.joblib
│   └── scaler_churn.joblib
├── notebooks/
│   ├── project_II_REGRESSION_rotemgibri.ipynb
│   └── project_II_CLASSIFICATION_rotemgibri.ipynb
├── .gitignore
└── README.md
```

---

## Part 1 — Regression: GDP Prediction

**Goal:** Predict a country's GDP based on COVID-19 statistics and economic indicators.

**Dataset:** Country-level data from 2021–2022 — confirmed cases, deaths, recoveries, unemployment rate, and CPI.

**Tools & Methods:**
- Data aggregation with `groupby()` — `sum` for COVID columns, `mean` for economic indicators
- `log1p` transformation to handle extreme value ranges
- Correlation heatmap to detect multicollinearity
- Models: Linear Regression, RidgeCV, LassoCV, Polynomial Regression
- Hyperparameter tuning with Cross Validation and GridSearch

**Results:**

| Model | MAE | MSE | RMSE |
|-------|-----|-----|------|
| Linear Regression | 1.10 | 1.97 | 1.40 |
| **Ridge ✅** | **1.06** | **1.85** | **1.36** |
| Lasso | 1.07 | 1.86 | 1.36 |
| Polynomial | 17.63+ | — | — |

**Key finding:** Strong multicollinearity between COVID features (Confirmed ↔ Deaths: 0.95) made Ridge the best choice. Polynomial Regression overfit severely, confirming the relationship between features and GDP is linear.

---

## Part 2 — Classification: Customer Churn Prediction

**Goal:** Predict whether a customer will cancel their subscription, enabling proactive retention.

**Dataset:** 64,374 customer records with behavioral and demographic features.

**Tools & Methods:**
- `get_dummies()` for categorical encoding
- Correlation heatmap + pairplot for feature analysis
- StandardScaler for feature scaling
- Models: Logistic Regression, KNN, SVM, Random Forest
- Hyperparameter tuning with GridSearchCV and elbow method

**Results:**

| Model | Accuracy | Recall | F1 |
|-------|----------|--------|----|
| Logistic Regression | 82.9% | 82.7% | 82.2% |
| KNN (K=10) | 91.3% | 91.8% | 90.9% |
| SVM (C=10, rbf) | 94.9% | 95.9% | 94.8% |
| **Random Forest ✅** | **99.95%** | **99.93%** | **99.96%** |

**Key finding:** Payment Delay (0.56) and Support Calls (0.30) were the strongest churn predictors. Random Forest significantly outperformed all other models with only 8 misclassifications out of ~19,000 test samples.

---

## What I Learned

- How to make data-driven decisions at every step — from choosing `sum` vs `mean` for aggregation, to selecting Ridge over Linear Regression based on detected multicollinearity
- The importance of removing irrelevant identifiers early — `CustomerID` caused a false correlation of 0.53 with Churn
- How to handle large datasets efficiently — used stratified sampling for KNN and SVM tuning
- How to validate that high training accuracy is not overfitting by verifying on the test set
- Full deployment workflow: training final models on entire dataset, exporting with joblib, and verifying reload

---

## Dependencies

```
pandas · numpy · scikit-learn · seaborn · matplotlib · joblib
```

---

*Rotem Gibri — ECOM School AI & Data Science Course*