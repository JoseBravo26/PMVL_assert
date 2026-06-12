import json
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path

# Dossier où seront stockés les logs de production
LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "prediction_logs.jsonl"

def log_prediction_event(inputs: dict, outputs: dict, status: str, latency_ms: float):
    """
    Enregistre les détails d'une requête de prédiction dans un fichier JSONL.
    Permet l'analyse ultérieure de la dérive (drift) et des performances.
    """
    # Créer le dossier s'il n'existe pas
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Créer l'événement structuré
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4()),
        "status": status,
        "latency_ms": round(latency_ms, 2),
        "inputs": inputs,
        "outputs": outputs,
    }
    
    # Ajouter la ligne au fichier log
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")