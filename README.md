# bike-sharing-
# 📊 MLflow Regression Models Dashboard (Streamlit)

An interactive **Streamlit web application** that loads multiple **MLflow-registered regression models**, evaluates them on a dataset, and visualizes their performance using standard regression metrics and plots.

This app allows you to **compare models in real time** directly from the MLflow Model Registry.

---

## 🚀 Features

- 🔗 Load models directly from **MLflow Model Registry**
- 📈 Compare multiple regression models
- 📊 View evaluation metrics:
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Accuracy-like percentage (R² × 100)
- 📉 Actual vs Predicted scatter plot
- 📋 Sample predictions table
- ⚡ Fast loading with Streamlit caching

---

## 🧠 Models Supported

The app looks for the following models in the MLflow registry:

- LinearRegression
- RandomForestRegressor
- GradientBoostingRegressor
- KNeighborsRegressor
- SVR
- LGBMRegressor

> ⚠️ If a model is not found in MLflow, a warning is shown in the UI.

---

## 📂 Dataset

- Example dataset: **Bike Sharing Dataset**
- Target column: `cnt`
- Features: all remaining columns



df = pd.read_csv("day.csv")


## 🛠️ Tech Stack

Python

Streamlit

MLflow

Scikit-learn

Pandas

NumPy

Matplotlib

## ▶️ How to Run the App

### 1️⃣ Install Dependencies
pip install -r requirements.txt

### 2️⃣ Start MLflow Tracking Server (if not running)
mlflow ui

### 3️⃣ Run the Streamlit App
streamlit run app.py

## 📊 App Preview

- Select a regression model from the sidebar

- View performance metrics instantly

- Analyze prediction accuracy visually

- Compare models efficiently

## 📌 Use Cases

- Model comparison & evaluation

- MLflow model registry visualization

- Regression performance analysis

- MLOps demos and portfolios
