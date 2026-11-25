from tests.conftest import auth_headers


def test_create_and_list_patient(client):
    headers = auth_headers(client, "boss")
    created = client.post("/patients", json={"name": "Asha", "age": 40}, headers=headers)
    assert created.status_code == 200

    patients = client.get("/patients", headers=headers).json()
    assert len(patients) == 1
    assert patients[0]["name"] == "Asha"


def test_doctor_cannot_see_another_doctors_patient(client):
    auth_headers(client, "boss")                      # admin, so doctors are doc1/doc2
    doc1 = auth_headers(client, "doc1")
    doc2 = auth_headers(client, "doc2")

    patient = client.post("/patients", json={"name": "Asha", "age": 40}, headers=doc1).json()

    assert client.get("/patients", headers=doc2).json() == []
    response = client.get(f"/patients/{patient['id']}/predictions", headers=doc2)
    assert response.status_code == 404


def test_admin_sees_all_patients(client):
    admin = auth_headers(client, "boss")
    doc1 = auth_headers(client, "doc1")
    client.post("/patients", json={"name": "Asha", "age": 40}, headers=doc1)

    assert len(client.get("/patients", headers=admin).json()) == 1