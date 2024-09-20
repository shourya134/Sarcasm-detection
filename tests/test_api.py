import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api import MODEL_PATH, VECTORIZER_PATH

ARTIFACTS_AVAILABLE = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_AVAILABLE, reason="artifacts/model.joblib and artifacts/vectorizer.joblib not found — run train.py first"
)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from src.api import app

    return TestClient(app)


def test_predict_returns_expected_shape(client):
    response = client.post("/predict", json={"headline": "Local Man Discovers Sarcasm, Nation Stunned"})

    assert response.status_code == 200
    body = response.json()
    assert "is_sarcastic" in body
    assert "confidence" in body
    assert isinstance(body["is_sarcastic"], bool)


def test_predict_confidence_in_valid_range(client):
    response = client.post("/predict", json={"headline": "Federal Reserve Raises Interest Rates"})

    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 1.0


def test_predict_rejects_missing_headline(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422
