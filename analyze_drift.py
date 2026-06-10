import pandas as pd
import json
import numpy as np
from pathlib import Path

# Imports pour Evidently (Détection de Drift)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

# ==========================================
# 1. Configuration des chemins
# ==========================================
REFERENCE_DATA_PATH = "pmvl_cleaned_prepared_with_features-3.csv"
LOGS_DATA_PATH = "logs/inference_results.jsonl"
PRODUCTION_LOGS_PATH = "logs/production_logs.jsonl"
OUTPUT_REPORT_PATH = "data_drift_report.html"

# ==========================================
# 2. Analyse Opérationnelle (Latence, Erreurs)
# ==========================================
print("--- ANALYSE OPÉRATIONNELLE ---")
if Path(PRODUCTION_LOGS_PATH).exists():
    logs_df = pd.read_json(PRODUCTION_LOGS_PATH, lines=True)
    
    total_requests = len(logs_df)
    error_rate = (logs_df['status_code'] >= 400).mean() * 100
    avg_latency = logs_df['latency_ms'].mean()
    p95_latency = logs_df['latency_ms'].quantile(0.95)
    
    print(f"Total requêtes : {total_requests}")
    print(f"Taux d'erreur : {error_rate:.2f}%")
    print(f"Latence Moyenne : {avg_latency:.2f} ms")
    print(f"Latence P95 : {p95_latency:.2f} ms\n")
else:
    print(f"Fichier {PRODUCTION_LOGS_PATH} introuvable pour les métriques.")

# ==========================================
# 3. Préparation des données pour le Drift
# ==========================================
print("--- ANALYSE DE DATA DRIFT ---")
# 3a. Chargement des données de référence (Entraînement)
print("Chargement des données de référence...")
reference_df = pd.read_csv(REFERENCE_DATA_PATH)

# Les colonnes que notre modèle utilise (à adapter si besoin)
feature_columns = [
    "holding_date", "pmvl_estim", "quantity", "purch_val_clean", "quote",
    "vnc_agrege_dirty", "entite", "isin", "orig_name", "ticker", 
    "ref_unik_asset", "fund_code", "col_3a", "canton"
]

# On garde uniquement les features
ref_data = reference_df[feature_columns].copy()

# 3b. Chargement des données de production (Inférence)
print("Chargement des données de production...")
if not Path(LOGS_DATA_PATH).exists():
    print(f"ERREUR: Fichier {LOGS_DATA_PATH} introuvable. Faites quelques requêtes sur l'API d'abord.")
    exit()

prod_records = []
with open(LOGS_DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        # On extrait les inputs
        record = data.get("input_features", {})
        prod_records.append(record)

prod_data = pd.DataFrame(prod_records)

# On s'assure que les colonnes correspondent
common_cols = list(set(ref_data.columns).intersection(set(prod_data.columns)))
ref_data = ref_data[common_cols]
prod_data = prod_data[common_cols]

# ==========================================
# 4. Génération du rapport avec Evidently
# ==========================================
print("Génération du rapport Evidently...")

# On configure un rapport Evidently avec le preset de Data Drift
drift_report = Report(metrics=[
    DataDriftPreset(),
])

# Calcul du drift entre la référence (entraînement) et la prod
drift_report.run(reference_data=ref_data, current_data=prod_data)

# Sauvegarde du rapport en HTML
drift_report.save_html(OUTPUT_REPORT_PATH)
print(f"✅ Analyse terminée ! Rapport sauvegardé dans : {OUTPUT_REPORT_PATH}")
print(f"Ouvrez ce fichier dans votre navigateur pour visualiser les résultats.")