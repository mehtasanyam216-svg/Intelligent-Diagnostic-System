import joblib
import pandas as pd
import warnings

# This just hides some annoying background warnings pandas likes to throw
warnings.filterwarnings('ignore') 

print("🧠 Waking up the AI Brain...")
# 1. Load the saved model (from our models folder)
model = joblib.load('../models/diabetes_rf_model.pkl')
print("✅ AI loaded successfully!\n")

# 2. Create a brand new patient profile
print("--- NEW PATIENT VITALS ---")
print("Glucose: 165 | BMI: 35.0 | Age: 42")

# We format it exactly how the AI expects (the 8 columns)
new_patient = pd.DataFrame([[3, 165, 80, 25, 110, 35.0, 0.8, 42]], 
                           columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

# 3. Ask the AI to make a prediction
prediction = model.predict(new_patient)[0]

# 4. Print the final diagnosis
print("\n===============================")
if prediction == 1:
    print(" ⚠️ DIAGNOSIS: HIGH RISK")
else:
    print(" ✅ DIAGNOSIS: LOW RISK")
print("===============================\n")