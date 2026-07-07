Intelligent Diagnostic Systems 🏥🧠
An automated, dual-architecture Artificial Intelligence healthcare platform designed to assist medical professionals by bridging the gap between raw medical data and actionable clinical insights.

This repository contains two distinct diagnostic engines: a Classical Machine Learning model for tabular physiological data, and a Deep Learning Convolutional Neural Network for spatial MRI image classification. Both models are deployed via interactive web dashboards.

📑 Table of Contents
Project 1: Diabetes Risk Assessment

Project 2: NeuroScan Brain Tumor Detection

Technology Stack

Local Installation & Setup

Author

🩸 Project 1: Diabetes Risk Assessment (Tabular Engine)
A stateless diagnostic pipeline that predicts a patient's risk of developing diabetes based on 8 physiological vitals.

Dataset: Pima Indians Diabetes Database (768 patient records).

Algorithm: Random Forest Classifier.

Why Random Forest? Chosen for its robustness against overfitting on small tabular datasets and its ability to handle non-linear relationships natively.

Key Features:

Real-time risk confidence scoring.

Robust preprocessing handling invalid zero-values via median imputation.

Automated generation of downloadable FPDF clinical reports.

🧠 Project 2: NeuroScan Brain Tumor Detection (Vision Engine)
A computer vision application utilizing Deep Learning to classify unstructured MRI spatial data into four distinct tumor categories (Glioma, Meningioma, Pituitary, or No Tumor).

Dataset: Kaggle MRI Dataset (~3,200 images).

Algorithm: MobileNetV2 (Convolutional Neural Network).

Training Methodology: * Transfer Learning: Base layers frozen to extract generic spatial features.

Selective Fine-Tuning: Top 20 layers unfrozen to specialize the network specifically for MRI textures.

Key Features:

High-efficiency depthwise separable convolutions for low-latency web inference.

Automatic image preprocessing (224x224 resizing, pixel normalization, and batch dimension expansion).

Structural model reconstruction and weight-injection to bypass strict Keras metadata versioning conflicts.

💻 Technology Stack
Frontend: Streamlit, HTML/CSS (via Markdown)

Backend: Python 3.10+

Classical ML: Scikit-Learn, Pandas, Joblib

Deep Learning: TensorFlow, Keras (MobileNetV2)

Image Processing: Pillow (PIL), NumPy

Utilities: FPDF (Report Generation), Base64

⚙️ Local Installation & Setup
Follow these steps to run the diagnostic dashboards on your local machine.

1. Clone the repository

Bash
git clone https://github.com/yourusername/Intelligent-Diagnostic-Systems.git
cd Intelligent-Diagnostic-Systems
2. Install Dependencies
Ensure you have Python 3.10+ installed, then run:

Bash
pip install streamlit pandas scikit-learn joblib tensorflow pillow numpy fpdf
3. Run the Diabetes Dashboard

Bash
cd Diabetes-Assessment
streamlit run app.py
4. Run the NeuroScan Dashboard

Bash
cd NeuroScan-App
streamlit run app.py
(Note: Ensure your diabetes_rf_model.pkl and brain_tumor_final.keras files are placed in their respective project directories before running).

👨‍💻 Author
Sanyam Kumar

Information Technology (B.Tech)

Maharaja Surajmal Institute of Technology (MSIT), Delhi

Email: mehtasanyam216@gmail.com
