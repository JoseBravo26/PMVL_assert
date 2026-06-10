from fastapi.testclient import TestClient
from app.main import app

# Création du client de test FastAPI
client = TestClient(app)

# Payload valide (contenant tous les champs obligatoires définis dans schemas.py)
# On utilise ici les noms des variables Python (grâce à populate_by_name=True)
VALID_PAYLOAD = {
    "holding_date": "2026-03-01",
    "pmvl_estim": 1500.50,
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
    """
    Test 1 : Vérifie que le endpoint de diagnostic répond bien 200 OK.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_data():
    """
    Test 2 : Vérifie qu'un payload complet et valide renvoie bien une prédiction.
    """
    response = client.post("/predict", json=VALID_PAYLOAD)
    
    # Ligne ajoutée pour debug
    print("RESPONSE STATUS:", response.status_code)
    print("RESPONSE BODY:", response.json())

    assert response.status_code == 200
    
    # Vérifie la structure de la réponse
    data = response.json()
    assert "proba_bonne_estimation" in data
    assert "prediction" in data
    assert "seuil_applique" in data
    assert data["fund_code"] == "FUND_001"
    
    # La probabilité doit être entre 0 et 1
    assert 0.0 <= data["proba_bonne_estimation"] <= 1.0


def test_predict_missing_required_field():
    """
    Test 3 : Vérifie que l'API renvoie une erreur 422 si un champ obligatoire (ex: quantity) est manquant.
    """
    invalid_payload = VALID_PAYLOAD.copy()
    del invalid_payload["quantity"]  # On supprime un champ obligatoire

    response = client.post("/predict", json=invalid_payload)
    
    # 422 Unprocessable Entity est le code standard de FastAPI pour une erreur de validation
    assert response.status_code == 422


def test_predict_invalid_data_type():
    """
    Test 4 : Vérifie que l'API renvoie une erreur 422 si un type de donnée est incorrect 
    (ex: une chaîne de caractères au lieu d'un float pour 'quote').
    """
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["quote"] = "Ceci_n_est_pas_un_chiffre"

    response = client.post("/predict", json=invalid_payload)
    
    assert response.status_code == 422