# 💼 Salary Prediction Pipeline (ML Regression Project)

This project presents a machine learning pipeline for predicting employee salaries using various features related to job role, experience, and company attributes. It explores and compares multiple regression models, incorporates hyperparameter tuning using Bayesian Optimization, and evaluates model performance using standard regression metrics. Feature importance is also analyzed to understand the most influential predictors.

---

## 📂 Files in This Repo

- `Salary_prediction_pipeline.ipynb` — Jupyter notebook with the complete ML pipeline.
- `ml_salary_dataset_1000.csv` — Cleaned dataset with 1000 samples used for training and evaluation.
- `requirements.txt` — List of Python dependencies needed to run the notebook.

---

## 📊 Dataset Overview

- **Filename:** `ml_salary_dataset_1000.csv`
- **Rows:** 1000  
- **Target Variable:** `Salary (USD)`  
- **Feature Types:**
  - **Numerical:**  
    - `Experience (years)`  
    - `Years at Current Company`  
    - `Remote Ratio`  
  - **Categorical:**  
    - `Location`  
    - `Education Level`  
    - `Company Size`  
    - `Company Tier`  

---

## 🧠 Models Evaluated

The following regression models are trained and evaluated:

- **Linear Regression** — A baseline model for comparison.
- **Random Forest Regressor** — Ensemble model with hyperparameter tuning using `BayesSearchCV`.
- **Support Vector Regressor (SVR)** — Evaluated with different kernels (`linear`, `rbf`, etc.).

---

## ⚙️ Hyperparameter Tuning

- **RandomForestRegressor:** Optimized using `BayesSearchCV` from `scikit-optimize` (`skopt`) for improved performance.
- **SVR:** Grid search is used to tune kernel-specific parameters.

---

## 📈 Evaluation Metrics

Each model is assessed using:

- ✅ R² Score (Coefficient of Determination)
- 🔻 Mean Absolute Error (MAE)
- 🔺 Mean Squared Error (MSE)

The model with the best **R² score** is selected for further analysis.

---

## 🔍 Feature Importance

If `SVR` is the best-performing model, permutation-based feature importance is used to visualize the top 15 most influential features.

---

## 🛠 How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Launch the Notebook

Open the Jupyter notebook and run each cell in sequence:

```bash
jupyter notebook Salary_prediction_pipeline.ipynb
```

Ensure the dataset ml_salary_dataset_1000.csv is in the same directory as the notebook.




