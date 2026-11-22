import io
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from database import Base, engine, get_db
from models import Patient, Prediction

app = FastAPI(title="Intelligent Diagnostic API")

# Creates the tables in diagnostic.db if they don't exist yet
Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------
# Load both models ONCE when the server starts
# ---------------------------------------------------------------
diabetes_model = joblib.load("models/diabetes_rf_model.pkl")

FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def load_tumor_model():
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(4, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=out)
    model.load_weights("models/brain_tumor_final.keras")
    return model


tumor_model = load_tumor_model()
TUMOR_CLASSES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]


# ---------------------------------------------------------------
# Data shapes (input and output)
# ---------------------------------------------------------------
class DiabetesInput(BaseModel):
    Pregnancies: int = Field(ge=0, le=20)
    Glucose: float = Field(ge=0, le=250)
    BloodPressure: float = Field(ge=0, le=150)
    SkinThickness: float = Field(ge=0, le=100)
    Insulin: float = Field(ge=0, le=900)
    BMI: float = Field(ge=0, le=70)
    DiabetesPedigreeFunction: float = Field(ge=0, le=3)
    Age: int = Field(ge=21, le=120)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pregnancies": 1, "Glucose": 120, "BloodPressure": 70,
                "SkinThickness": 20, "Insulin": 79, "BMI": 25.0,
                "DiabetesPedigreeFunction": 0.5, "Age": 33,
            }
        }
    }


class DiabetesOutput(BaseModel):
    prediction: int
    diagnosis: str
    risk_confidence: float  # percentage


class TumorOutput(BaseModel):
    result: str
    confidence: float  # percentage


class PatientCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=120)


class PatientOut(BaseModel):
    id: int
    name: str
    age: int
    model_config = {"from_attributes": True}


class PredictionOut(BaseModel):
    id: int
    model_type: str
    result: str
    confidence: float
    created_at: datetime
    model_config = {"from_attributes": True}


# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------
def get_patient_or_404(db: Session, patient_id: int) -> Patient:
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Diagnostic API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/patients", response_model=PatientOut)
def create_patient(data: PatientCreate, db: Session = Depends(get_db)):
    patient = Patient(name=data.name, age=data.age)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@app.get("/patients", response_model=list[PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return db.query(Patient).all()


@app.get("/patients/{patient_id}/predictions", response_model=list[PredictionOut])
def patient_history(patient_id: int, db: Session = Depends(get_db)):
    get_patient_or_404(db, patient_id)
    return (
        db.query(Prediction)
        .filter(Prediction.patient_id == patient_id)
        .order_by(Prediction.created_at.desc())
        .all()
    )


@app.post("/predict/diabetes", response_model=DiabetesOutput)
def predict_diabetes(
    data: DiabetesInput,
    patient_id: int | None = None,
    db: Session = Depends(get_db),
):
    if patient_id is not None:
        get_patient_or_404(db, patient_id)

    patient_row = pd.DataFrame([data.model_dump()], columns=FEATURE_COLUMNS)
    prediction = int(diabetes_model.predict(patient_row)[0])
    risk = round(float(diabetes_model.predict_proba(patient_row)[0][1]) * 100, 1)
    diagnosis = "HIGH RISK" if prediction == 1 else "LOW RISK"

    if patient_id is not None:
        db.add(Prediction(
            patient_id=patient_id,
            model_type="diabetes",
            result=diagnosis,
            confidence=risk,
            details=json.dumps(data.model_dump()),
        ))
        db.commit()

    return DiabetesOutput(prediction=prediction, diagnosis=diagnosis, risk_confidence=risk)


@app.post("/predict/brain-tumor", response_model=TumorOutput)
async def predict_brain_tumor(
    file: UploadFile = File(...),
    patient_id: int | None = None,
    db: Session = Depends(get_db),
):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Please upload a JPG or PNG image")
    if patient_id is not None:
        get_patient_or_404(db, patient_id)

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
    img_batch = (np.asarray(image) / 255.0)[np.newaxis, ...]

    prediction = tumor_model.predict(img_batch, verbose=0)
    index = int(np.argmax(prediction[0]))
    result = TUMOR_CLASSES[index]
    confidence = round(float(prediction[0][index]) * 100, 2)

    if patient_id is not None:
        db.add(Prediction(
            patient_id=patient_id,
            model_type="brain_tumor",
            result=result,
            confidence=confidence,
        ))
        db.commit()

    return TumorOutput(result=result, confidence=confidence)