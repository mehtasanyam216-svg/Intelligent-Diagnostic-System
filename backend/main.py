import io
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from database import Base, engine, get_db
from models import Patient, Prediction, User
from security import (
    create_access_token,
    get_current_user,
    hash_password,
    require_admin,
    verify_password,
)

app = FastAPI(title="Intelligent Diagnostic API")

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
# Data shapes
# ---------------------------------------------------------------
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=72)


class UserOut(BaseModel):
    id: int
    username: str
    role: str
    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str


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
    risk_confidence: float


class TumorOutput(BaseModel):
    result: str
    confidence: float


class PatientCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=120)


class PatientOut(BaseModel):
    id: int
    name: str
    age: int
    doctor_id: int
    model_config = {"from_attributes": True}


class PredictionOut(BaseModel):
    id: int
    model_type: str
    result: str
    confidence: float
    created_at: datetime
    model_config = {"from_attributes": True}


# ---------------------------------------------------------------
# Helper: doctors can only touch their own patients, admins any
# ---------------------------------------------------------------
def get_patient_or_404(db: Session, patient_id: int, user: User) -> Patient:
    patient = db.get(Patient, patient_id)
    if patient is None or (user.role != "admin" and patient.doctor_id != user.id):
        # Same 404 for "doesn't exist" and "not yours", so we don't leak which IDs exist
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


# ---------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Diagnostic API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------
@app.post("/auth/register", response_model=UserOut)
def register(data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    # Simple bootstrap: the very first account becomes the admin, the rest are doctors
    role = "admin" if db.query(User).count() == 0 else "doctor"

    user = User(
        username=data.username,
        hashed_password=hash_password(data.password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return Token(
        access_token=create_access_token(user.username, user.role),
        token_type="bearer",
    )


@app.get("/auth/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@app.get("/admin/users", response_model=list[UserOut])
def list_users(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return db.query(User).all()


# ---------------------------------------------------------------
# Patient endpoints (login required)
# ---------------------------------------------------------------
@app.post("/patients", response_model=PatientOut)
def create_patient(
    data: PatientCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    patient = Patient(name=data.name, age=data.age, doctor_id=user.id)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@app.get("/patients", response_model=list[PatientOut])
def list_patients(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    query = db.query(Patient)
    if user.role != "admin":
        query = query.filter(Patient.doctor_id == user.id)
    return query.all()


@app.get("/patients/{patient_id}/predictions", response_model=list[PredictionOut])
def patient_history(
    patient_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_patient_or_404(db, patient_id, user)
    return (
        db.query(Prediction)
        .filter(Prediction.patient_id == patient_id)
        .order_by(Prediction.created_at.desc())
        .all()
    )


# ---------------------------------------------------------------
# Prediction endpoints (login required)
# ---------------------------------------------------------------
@app.post("/predict/diabetes", response_model=DiabetesOutput)
def predict_diabetes(
    data: DiabetesInput,
    patient_id: int | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if patient_id is not None:
        get_patient_or_404(db, patient_id, user)

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
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Please upload a JPG or PNG image")
    if patient_id is not None:
        get_patient_or_404(db, patient_id, user)

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