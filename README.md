# 💼 Salary Prediction Pipeline (ML Regression Project)

This project is a machine learning pipeline for predicting salaries based on a variety of employee and company attributes. It compares multiple regression models and includes hyperparameter tuning, performance evaluation, and feature importance visualization.

---

## 📂 Files in This Repo

- `Salary_prediction_pipeline.ipynb` — Main notebook containing the full end-to-end pipeline.
- `ml_salary_dataset_1000.csv` — Dataset used for model training and evaluation.
- `requirements.txt` — All Python packages required to run this notebook.
- *(Optional)*: `salary_model.pkl` — Trained model file (for deployment or API use).

---

## 📊 Dataset Overview

- File: `ml_salary_dataset_1000.csv`
- Target variable: `Salary (USD)`
- Feature types:
  - **Numerical**: `Experience (years)`, `Years at Current Company`, `Remote Ratio`
  - **Categorical**: `Location`, `Education Level`, `Company Size`, `Company Tier`

---

## 🧠 Models Evaluated

- `LinearRegression`
- `RandomForestRegressor` (Bayesian optimization via `BayesSearchCV`)
- `SVR` (Support Vector Regressor with multiple kernels)

---

## 📈 Metrics Used

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

The model with the highest R² score is selected and analyzed for feature importance using permutation importance.

---

## 🔍 Feature Importance

If `SVR` is the best model, the top 15 important features are visualized using permutation-based importance.

---

## 🛠 How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
