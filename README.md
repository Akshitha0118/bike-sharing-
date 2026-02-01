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

You can replace the dataset path with your own CSV file.

```python
df = pd.read_csv("path/to/your/dataset.csv")
