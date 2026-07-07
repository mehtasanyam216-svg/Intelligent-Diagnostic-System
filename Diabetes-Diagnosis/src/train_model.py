import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("🚀 Starting the Machine Learning Pipeline...")

# 1. Load Data
print("[1/5] Loading data...")
df = pd.read_csv('../data/diabetes.csv')

# 2. Clean Data (Fixing impossible zeros)
print("[2/5] Cleaning data...")
bad_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[bad_columns] = df[bad_columns].replace(0, np.nan)
for col in bad_columns:
    df[col] = df[col].fillna(df[col].median())

# 3. Split Data
print("[3/5] Splitting data into training and testing sets...")
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
print("[4/5] Training the Random Forest AI...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Save the AI's "Brain"
print("[5/5] Saving the trained model...")
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/diabetes_rf_model.pkl')

print("\n✅ Pipeline Complete! The AI brain has been updated and saved.")