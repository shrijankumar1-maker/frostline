import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_health_is_ok(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_predict_returns_valid_probability_and_threshold(client):
    reading = {
        "air_temp_c": 1.5,
        "dew_point_c": -1.0,
        "wind_speed_ms": 0.8,
        "cloud_cover_pct": 10,
    }

    response = client.post("/predict", json=reading)

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["frost_probability"] <= 1
    assert data["threshold"] == 0.35


def test_frosty_night_scores_higher_than_mild_one(client):
    frosty = {
        "air_temp_c": 1.5,
        "dew_point_c": -1.0,
        "wind_speed_ms": 0.8,
        "cloud_cover_pct": 10,
    }

    mild = {
        "air_temp_c": 9.0,
        "dew_point_c": 4.0,
        "wind_speed_ms": 3.5,
        "cloud_cover_pct": 80,
    }

    frosty_response = client.post("/predict", json=frosty)
    mild_response = client.post("/predict", json=mild)

    assert frosty_response.status_code == 200
    assert mild_response.status_code == 200

    frosty_data = frosty_response.json()
    mild_data = mild_response.json()

    assert frosty_data["frost_probability"] > mild_data["frost_probability"]

    assert frosty_data["alert"] is True
    assert mild_data["alert"] is False


def test_impossible_reading_is_rejected(client):
    invalid_reading = {
        "air_temp_c": 1.5,
        "dew_point_c": -1.0,
        "wind_speed_ms": -2.0,
        "cloud_cover_pct": 80,
    }

    response = client.post("/predict", json=invalid_reading)

    assert response.status_code == 422
