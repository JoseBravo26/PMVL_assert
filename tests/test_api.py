from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch
from app.main import app, run_model_prediction
from app.schemas import PMVLFeatures

client = TestClient(app)

# Données valides pour tester le modèle
VALID_PAYLOAD = {
    "holding_date": "2026-03-01",
    "pmvl_estim": 1500.0,
    "quantity": 100.0,
    "purch_val_clean": 45000.0,
    "quote": 510.0,
    "vnc_agrege_dirty": 49000.0,
    "entite": "ENTITE_TEST",
    "isin": "FR0000000001",
    "orig_name": "Asset Name Test",
    "ticker": "TICKER_TEST",
    "ref_unik_asset": "REF_12345",
    "fund_code": "FUND_001",
    "col_3a": "3A_TEST",
    "canton": "CANTON_TEST",
    "cic": "CIC_TEST",
    "groupe": "GROUPE_TEST",
    "ptf_name": "PTF_TEST"
}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "L'API PMVL est opérationnelle."}

def test_predict_success():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "proba_bonne_estimation" in data

def test_predict_missing_field():
    incomplete_payload = {"pmvl_estim": 1500.0, "quantity": 100.0}
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422

def test_download_logs():
    response = client.get("/download-logs")
    assert response.status_code == 200

def test_download_prod_logs():
    response = client.get("/download-prod-logs")
    assert response.status_code == 200

def test_run_model_prediction_direct():
    """Teste la fonction métier centrale directement pour la couverture"""
    features = PMVLFeatures(**VALID_PAYLOAD)
    result = run_model_prediction(features)
    assert hasattr(result, "proba_bonne_estimation")
    assert isinstance(result.prediction, bool)

@patch('app.main.run_model_prediction')
def test_predict_internal_error(mock_run_model):
    """Simule un crash du modèle pour tester le bloc except"""
    mock_run_model.side_effect = Exception("Erreur simulée pour les tests")
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 500
    assert "Erreur simulée pour les tests" in response.json()["detail"]
