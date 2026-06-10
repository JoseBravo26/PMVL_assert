import os
from functools import lru_cache
from catboost import CatBoostClassifier

# Variables d'environnement pour rendre le chemin dynamique (utile pour Docker)
MODEL_PATH_ENV = "MODEL_PATH"
DEFAULT_MODEL_PATH = "models/pmvl_catboost_final.cbm"

FEATURES_PATH_ENV = "FEATURES_PATH"
DEFAULT_FEATURES_PATH = "models/pmvl_feature_columns.txt"

@lru_cache(maxsize=1)
def get_model() -> CatBoostClassifier:
    """
    Charge le modèle CatBoost depuis le disque.
    Grâce à @lru_cache, cette fonction ne s'exécute réellement qu'une seule fois.
    """
    model_path = os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier du modèle est introuvable au chemin : {model_path}")
    
    print("Chargement du modèle CatBoost en mémoire...")
    model = CatBoostClassifier()
    model.load_model(model_path)
    print("Modèle chargé avec succès !")
    
    return model

#chargeur pour la liste de colonnes utilisées par le modèle (pour s'assurer que l'ordre des features est correct)
@lru_cache(maxsize=1)
def get_feature_columns() -> list[str]:
    features_path = os.getenv(FEATURES_PATH_ENV, DEFAULT_FEATURES_PATH)
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Le fichier des features est introuvable au chemin : {features_path}")
    with open(features_path, encoding="utf-8") as f:
        cols = [line.strip() for line in f if line.strip()]
    return cols