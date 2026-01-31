import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="MLflow Regression Models",
    layout="wide"
)

st.title("📊 MLflow Registered Regression Models")

# -----------------------------
# Load Data (replace with yours)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Admin\Desktop\bike sharing project\day.csv")   # <-- your dataset
    X = df.drop("cnt", axis=1)
    y = df["cnt"]
    return X, y

X_test, y_test = load_data()

# -----------------------------
# Load Models from MLflow
# -----------------------------
@st.cache_resource
def load_models():
    model_names = [
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "KNeighborsRegressor",
        "SVR",
        "LGBMRegressor"
    ]

    models = {}
    for name in model_names:
        try:
            models[name] = mlflow.sklearn.load_model(
                f"models:/{name}/latest"
            )
        except:
            st.warning(f"⚠️ {name} not found in MLflow registry")

    return models

models = load_models()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🔍 Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose a Model",
    list(models.keys())
)

model = models[selected_model_name]

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
accuracy_like = r2 * 100

# -----------------------------
# Display Metrics
# -----------------------------
st.subheader(f"📈 Evaluation – {selected_model_name}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("R² Score", f"{r2:.4f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("MSE", f"{mse:.2f}")
col4.metric("Accuracy-like %", f"{accuracy_like:.2f}%")

# -----------------------------
# Actual vs Predicted Plot
# -----------------------------
st.subheader("📊 Actual vs Predicted")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title(f"{selected_model_name}")

st.pyplot(fig)

# -----------------------------
# Results Table (Optional)
# -----------------------------
st.subheader("📋 Prediction Sample")

results_df = pd.DataFrame({
    "Actual": y_test.values[:20],
    "Predicted": y_pred[:20]
})

st.dataframe(results_df)
