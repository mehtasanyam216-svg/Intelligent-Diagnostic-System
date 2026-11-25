from tests.conftest import auth_headers


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_first_user_is_admin_second_is_doctor(client):
    first = client.post("/auth/register", json={"username": "boss", "password": "password123"})
    second = client.post("/auth/register", json={"username": "doc1", "password": "password123"})
    assert first.json()["role"] == "admin"
    assert second.json()["role"] == "doctor"


def test_duplicate_username_rejected(client):
    client.post("/auth/register", json={"username": "boss", "password": "password123"})
    again = client.post("/auth/register", json={"username": "boss", "password": "password123"})
    assert again.status_code == 400


def test_short_password_rejected(client):
    response = client.post("/auth/register", json={"username": "boss", "password": "short"})
    assert response.status_code == 422


def test_wrong_password_rejected(client):
    client.post("/auth/register", json={"username": "boss", "password": "password123"})
    response = client.post("/auth/login", data={"username": "boss", "password": "wrong-password"})
    assert response.status_code == 401


def test_protected_endpoint_needs_token(client):
    assert client.get("/patients").status_code == 401


def test_admin_endpoint_blocks_doctors(client):
    admin = auth_headers(client, "boss")      # first user = admin
    doctor = auth_headers(client, "doc1")
    assert client.get("/admin/users", headers=admin).status_code == 200
    assert client.get("/admin/users", headers=doctor).status_code == 403