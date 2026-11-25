import os

# Must be set BEFORE importing the app, so tests use a throwaway database
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["SECRET_KEY"] = "test-secret-key"

import pytest
from fastapi.testclient import TestClient

from database import Base, engine
from main import app


@pytest.fixture(autouse=True)
def clean_database():
    """Gives every test a fresh, empty database."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client():
    return TestClient(app)


def auth_headers(client, username, password="password123"):
    """Registers a user (if new), logs in, and returns the Authorization header."""
    client.post("/auth/register", json={"username": username, "password": password})
    response = client.post(
        "/auth/login", data={"username": username, "password": password}
    )
    return {"Authorization": f"Bearer {response.json()['access_token']}"}