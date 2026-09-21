import pytest
from fastapi.testclient import TestClient

from api.database import Base, engine
from api.main import app

SAMPLE_HOUSE = {
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 1800,
    "grade": 8,
    "zipcode": "98103",
}


@pytest.fixture
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(bind=engine)


def test_index_reports_ok_status(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_and_get_house(client):
    created = client.post("/houses", json=SAMPLE_HOUSE)
    assert created.status_code == 201
    house_id = created.json()["id"]

    fetched = client.get(f"/houses/{house_id}")
    assert fetched.status_code == 200
    assert fetched.json()["zipcode"] == "98103"


def test_get_all_houses_returns_created_houses(client):
    client.post("/houses", json=SAMPLE_HOUSE)
    client.post("/houses", json={**SAMPLE_HOUSE, "zipcode": "98004"})

    response = client.get("/houses")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_update_house_replaces_editable_fields(client):
    house_id = client.post("/houses", json=SAMPLE_HOUSE).json()["id"]

    updated = client.put(
        f"/houses/{house_id}", json={**SAMPLE_HOUSE, "bedrooms": 5}
    )

    assert updated.status_code == 200
    assert updated.json()["bedrooms"] == 5


def test_delete_house_removes_it(client):
    house_id = client.post("/houses", json=SAMPLE_HOUSE).json()["id"]

    deleted = client.delete(f"/houses/{house_id}")
    assert deleted.status_code == 204

    missing = client.get(f"/houses/{house_id}")
    assert missing.status_code == 404


def test_get_missing_house_returns_404(client):
    response = client.get("/houses/999999")

    assert response.status_code == 404


def test_create_house_rejects_invalid_grade(client):
    response = client.post("/houses", json={**SAMPLE_HOUSE, "grade": 99})

    assert response.status_code == 422
