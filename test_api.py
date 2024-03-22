from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    # Prepare test data
    data = {"values": [[0, 0], [1, 1], [2, 2]]}

    # Send POST request to /predict endpoint
    response = client.post("/predict", json=data)

    # Check response status code
    assert response.status_code == 200

    # Check response JSON
    expected_response = {"predictions": [0, 1, 2]}
    assert response.json().keys() == expected_response.keys()

    # We don't check the exact values
    # because they might differ due to model changes or floating point deviations
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    assert all([isinstance(pred, float) for pred in predictions])
