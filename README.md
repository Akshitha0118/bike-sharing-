# 🚲 End-To-End-Bike-Rental-Management-System-MLOps


## 📌 Project Overview

The rapid growth of bike-sharing systems has led to the generation of large volumes of data related to **weather conditions, temporal patterns, and user behavior**. Accurately predicting bike rental demand is critical for:

- Optimizing bike distribution and resource allocation  
- Improving customer satisfaction  
- Reducing operational and maintenance costs  

This project builds an **end-to-end machine learning regression system** to predict the **total number of bike rentals (`cnt`)** using historical data and multiple influencing factors.

---

## 🎯 Problem Statement

Bike rental demand is affected by a combination of:

- 🌦 **Weather conditions**: temperature, humidity, windspeed  
- 📆 **Temporal patterns**: season, month, weekday, working day  
- 🎉 **External factors**: holidays, weather situations  

The challenge is to design a **robust regression-based ML pipeline** that can effectively learn these patterns and deliver accurate predictions.

---

## ✅ Project Objectives

- Perform **data preprocessing and exploratory data analysis (EDA)**
- Train and compare multiple **regression models**
- Evaluate models using multiple performance metrics
- Track experiments, metrics, and models using **MLflow**
- Register and load the **best-performing model**
- Build an **interactive Streamlit application** for predictions and visualization

---

## 🤖 Machine Learning Models Used

The following regression models are trained and evaluated:

- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- K-Nearest Neighbors (KNN) Regressor  
- Support Vector Regressor (SVR)  
- LightGBM Regressor (LGBM)  

---

## 📊 Model Evaluation Metrics

Each model is evaluated using:

- **R² Score**
- **Adjusted R²**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Accuracy-like Metric** → `R² × 100`

---

## 🧪 Experiment Tracking with MLflow

This project leverages **MLflow** to ensure reproducibility and experiment tracking:

- Logs model parameters
- Tracks evaluation metrics
- Stores trained models
- Registers the best-performing model
- Enables model loading for deployment

---

## 🌐 Streamlit Application

An interactive **Streamlit web app** is built to:

- Input environmental and temporal features
- Predict bike rental demand in real time
- Visualize predictions and model performance
- Load models directly from MLflow Registry

---
## ⚙️ Installation & Setup

#### 1️⃣ Clone the Repository
[git clone https://github.com/your-username/bike-demand-prediction.git](https://github.com/Akshitha0118/End-To-End-Bike-Rental-Management-System-MLOps)

---

#### 2️⃣ Install Dependencies
pip install -r requirements.txt
---

#### 3️⃣ Run MLflow UI
mlflow ui


Access at: http://localhost:5000
---
#### 4️⃣ Run the Streamlit App
streamlit run app.py
---

## 📈 Expected Outcomes

Identify the best-performing regression model for bike demand prediction

Understand feature importance impacting bike rentals

Enable reproducible ML workflows using MLflow

Deliver a deployable Streamlit interface for real-time predictions
---

## 🛠️ Tech Stack

Python

Scikit-learn

LightGBM

MLflow

Streamlit

Pandas, NumPy, Matplotlib, Seaborn
