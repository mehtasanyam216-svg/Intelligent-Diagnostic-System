import streamlit as st
import pandas as pd
import joblib
import time
from fpdf import FPDF

# 1. Setup the Page (Wide layout for a dashboard feel)
st.set_page_config(page_title="AI Risk Assessor", page_icon="🧬", layout="wide")

# PDF Generation Function (Unchanged)
def create_pdf(vitals, diagnosis, risk_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Automated AI Risk Assessment", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Diagnosis: {diagnosis}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Confidence: {risk_score:.1f}%", ln=True)
    pdf.ln(5)
    for key, value in vitals.items():
        pdf.cell(200, 8, txt=f"- {key}: {value}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('models/diabetes_rf_model.pkl')

model = load_model()

# --- SIDEBAR: Inputs ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864303.png", width=60) # Adds a little medical icon
    st.title("Patient Vitals")
    st.markdown("Adjust the sliders to simulate patient data.")
    
    preg = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 0, 250, 120)
    bp = st.slider("Blood Pressure", 0, 150, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin Level", 0, 900, 79)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("Pedigree Function", 0.0, 3.0, 0.5)
    age = st.slider("Age", 21, 120, 33)

# --- MAIN PAGE: Dashboard ---
st.title("🧬 Diabetes Risk Prediction Engine")
st.markdown("Review the patient's vitals below and initialize the AI diagnostic model.")

# Display Inputs as Sleek Metric Cards
st.subheader("Current Patient Profile")
m1, m2, m3, m4 = st.columns(4)
m1.metric(label="Glucose", value=f"{glucose} mg/dL")
m2.metric(label="BMI", value=f"{bmi}")
m3.metric(label="Blood Pressure", value=f"{bp} mmHg")
m4.metric(label="Age", value=f"{age} Yrs")

st.markdown("---")

# --- ACTION & RESULTS ---
if st.button("Initialize AI Analysis 🚀", type="primary", use_container_width=True):
    
    # Fake loading state for better UX
    with st.spinner("AI is analyzing patient data matrices..."):
        time.sleep(1.5) # Pauses for 1.5 seconds to feel "heavy"
    
    vitals_dict = {
        "Pregnancies": preg, "Glucose": glucose, "Blood Pressure": bp,
        "Skin Thickness": skin, "Insulin": insulin, "BMI": bmi,
        "Pedigree Function": dpf, "Age": age
    }
    
    patient_data = pd.DataFrame([list(vitals_dict.values())], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    prediction = model.predict(patient_data)[0]
    risk_score = model.predict_proba(patient_data)[0][1] * 100
    diagnosis = "HIGH RISK" if prediction == 1 else "LOW RISK"

    # Sleek Results Display
    st.subheader("Diagnostic Results")
    st.write(f"**Confidence Level:** {risk_score:.1f}%")
    st.progress(risk_score / 100)
    
    if prediction == 1:
        st.error(f"⚠️ **{diagnosis} DETECTED** - Immediate clinical follow-up recommended.")
    else:
        st.success(f"✅ **{diagnosis} DETECTED** - Patient metrics are within acceptable ranges.")

    # PDF Download
    st.markdown("<br>", unsafe_allow_html=True) # Adds a little spacing
    pdf_data = create_pdf(vitals_dict, diagnosis, risk_score)
    st.download_button(
        label="📥 Export Clinical Report (PDF)", 
        data=pdf_data, 
        file_name=f"Patient_Report_{age}.pdf", 
        mime="application/pdf"
    )