import io

from PIL import Image

from tests.conftest import auth_headers

VALID_VITALS = {
    "Pregnancies": 1, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
    "Insulin": 79, "BMI": 25.0, "DiabetesPedigreeFunction": 0.5, "Age": 33,
}


def test_diabetes_prediction(client):
    headers = auth_headers(client, "boss")
    response = client.post("/predict/diabetes", json=VALID_VITALS, headers=headers)
    body = response.json()
    assert response.status_code == 200
    assert body["diagnosis"] in ("HIGH RISK", "LOW RISK")
    assert 0 <= body["risk_confidence"] <= 100


def test_diabetes_rejects_invalid_input(client):
    headers = auth_headers(client, "boss")
    bad = {**VALID_VITALS, "Glucose": -5}
    assert client.post("/predict/diabetes", json=bad, headers=headers).status_code == 422


def test_prediction_is_saved_to_patient_history(client):
    headers = auth_headers(client, "boss")
    patient = client.post("/patients", json={"name": "Asha", "age": 40}, headers=headers).json()

    client.post(
        "/predict/diabetes", json=VALID_VITALS,
        params={"patient_id": patient["id"]}, headers=headers,
    )

    history = client.get(f"/patients/{patient['id']}/predictions", headers=headers).json()
    assert len(history) == 1
    assert history[0]["model_type"] == "diabetes"


def test_prediction_for_unknown_patient_is_404(client):
    headers = auth_headers(client, "boss")
    response = client.post(
        "/predict/diabetes", json=VALID_VITALS, params={"patient_id": 999}, headers=headers
    )
    assert response.status_code == 404


def test_prediction_needs_login(client):
    assert client.post("/predict/diabetes", json=VALID_VITALS).status_code == 401


def test_brain_tumor_rejects_non_image(client):
    headers = auth_headers(client, "boss")
    response = client.post(
        "/predict/brain-tumor",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
        headers=headers,
    )
    assert response.status_code == 400


def test_brain_tumor_prediction_on_image(client):
    headers = auth_headers(client, "boss")
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), color=(90, 90, 90)).save(buffer, format="PNG")

    response = client.post(
        "/predict/brain-tumor",
        files={"file": ("scan.png", buffer.getvalue(), "image/png")},
        headers=headers,
    )
    body = response.json()
    assert response.status_code == 200
    assert body["result"] in ("Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor")
    assert 0 <= body["confidence"] <= 100