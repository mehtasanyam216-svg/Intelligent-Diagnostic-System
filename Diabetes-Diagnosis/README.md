# 🩺 AI Diabetes Risk Predictor

## 📌 Project Overview
An end-to-end Machine Learning web application designed to predict the likelihood of a patient developing diabetes. The system utilizes diagnostic measurements (e.g., Glucose, BMI, Age) to provide instant, AI-driven risk assessments through an interactive Streamlit dashboard.

## 🚀 Features
* **Machine Learning Backend:** Powered by a `RandomForestClassifier` trained on the Pima Indians Diabetes Database.
* **Data Engineering:** Includes an automated pipeline for cleaning biological data and handling missing zero-values via median imputation.
* **Interactive UI:** A live Streamlit dashboard allowing users to adjust patient vitals and receive real-time predictions.
* **Data Visualization:** Utilizes Plotly to generate dynamic probability gauge charts for clear visual feedback.

## 📁 Project Structure
```text
diabetes-prediction/
│
├── data/               # Contains the diabetes.csv dataset
├── models/             # Contains the exported Random Forest brain (.pkl)
├── notebooks/          # Jupyter notebooks for Exploratory Data Analysis
│   └── explore.ipynb   
├── src/                # Standalone automated training pipeline
│   └── train_model.py  # Script to retrain and save the model
├── app.py              # The Streamlit web application frontend
└── README.md

🛠️ Tech Stack
Language: Python 3

Machine Learning: Scikit-Learn, Pandas, NumPy, Joblib

Frontend/UI: Streamlit, 

👨‍💻 Author
Sanyam Kumar
B.Tech Information Technology, MSIT


Take a look at the web app with the new dial chart. How does the dashboard look now?