from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import os
import pandas as pd
import numpy as np
import gradio as gr

from .schemas import PMVLFeatures, PredictionResponse
from .model_loader import get_model, get_feature_columns

import json
import time
from uuid import uuid4
from pathlib import Path

GLOBAL_THRESHOLD_ENV = "GLOBAL_THRESHOLD"
DEFAULT_THRESHOLD = 0.45

app = FastAPI(
    title="API de prédiction de la qualité PMVL",
    description="API exposant le modèle CatBoost pour estimer la précision des PMVL.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# On crée le dossier logs s'il n'existe pas
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
PRODUCTION_LOGS_FILE = LOG_DIR / "production_logs.jsonl"

@app.middleware("http")
async def production_logging_middleware(request: Request, call_next):
    """
    Middleware pour logger chaque requête à l'API en production.
    """
    start_time = time.time()
    request_id = str(uuid4())
    
    # ⚠️ MODIFICATION IMPORTANTE ICI ⚠️
    # On ne lit le body et on ne loggue QUE si c'est explicitement NOTRE endpoint d'API
    if request.url.path == "/predict" and request.method == "POST":
        body_bytes = await request.body()
        
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            error_msg = None
        except Exception as e:
            status_code = 500
            error_msg = str(e)
            raise e
        finally:
            latency_ms = (time.time() - start_time) * 1000
            
            # Extraction des inputs
            input_data = None
            if body_bytes:
                try:
                    input_data = json.loads(body_bytes.decode("utf-8"))
                except json.JSONDecodeError:
                    input_data = "Unparseable JSON"
            
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "request_id": request_id,
                "endpoint": request.url.path,
                "latency_ms": round(latency_ms, 2),
                "status_code": status_code,
                "input_features": input_data,
                "error": error_msg
            }
            
            with open(PRODUCTION_LOGS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        return response

    # ⚠️ Pour toutes les autres routes (incluant l'interface Gradio) ⚠️
    # On laisse passer la requête normalement sans lire le body pour ne pas bloquer Gradio
    else:
        return await call_next(request)

GROUP_KEYS = [
    "PMVL[ENTITE]",
    "PMVL[Selected Fund code]",
    "PMVL[ISIN]",
    "PMVL[Ref Unik Asset]",
]


def make_position_group(df: pd.DataFrame) -> pd.Series:
    """Recrée la colonne position_group comme dans le notebook"""
    existing = [c for c in GROUP_KEYS if c in df.columns]
    if not existing:
        return pd.Series(
            np.arange(len(df)).astype(str), index=df.index, name="position_group"
        )
    return (
        df[existing]
        .astype(str)
        .fillna("NA")
        .agg("||".join, axis=1)
        .rename("position_group")
    )


def prepare_catboost_features(X: pd.DataFrame, cat_cols: list):
    """Prépare les features (remplace les manquants pour les cat, etc.)"""
    X = X.copy()
    for c in X.columns:
        if c in cat_cols:
            # Remplacement des valeurs manquantes par 'MISSING' et conversion en string
            X[c] = X[c].fillna("MISSING").astype(str)
        else:
            # Remplacement par NaN et forçage numérique
            X[c] = pd.to_numeric(X[c], errors="coerce")

    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X


@app.on_event("startup")
def load_model_on_startup():
    try:
        get_model()
        get_feature_columns()
    except Exception as e:
        print(f"Erreur critique lors du chargement initial : {e}")


def run_model_prediction(features: PMVLFeatures) -> PredictionResponse:
    """
    Fonction centrale de prédiction réutilisée par l'API FastAPI et l'UI Gradio.
    """
    model = get_model()
    feature_columns = get_feature_columns()

    # 1) Récupérer les données brutes
    raw_dict = features.model_dump(by_alias=True, mode="json")
    # Le modèle n'utilise pas directement la date
    raw_dict.pop("PMVL[Holding date]", None)

    # 2) Construire une ligne avec toutes les colonnes attendues
    row = {col: np.nan for col in feature_columns}
    for k, v in raw_dict.items():
        if k in row and v is not None:
            row[k] = v

    raw_df = pd.DataFrame([row])

    # 3) Recréer position_group si elle fait partie des features
    if "position_group" in feature_columns:
        raw_df["position_group"] = make_position_group(raw_df)

    # 4) S'assurer de l'ordre exact des colonnes
    df_input = raw_df[feature_columns]

    # 5) Nettoyage en utilisant les colonnes catégorielles du modèle
    cat_indices = model.get_cat_feature_indices()
    cat_cols = [feature_columns[i] for i in cat_indices]
    df_input = prepare_catboost_features(df_input, cat_cols)

    # 6) Prédire
    proba = float(model.predict_proba(df_input)[:, 1][0])

    threshold = float(os.getenv(GLOBAL_THRESHOLD_ENV, DEFAULT_THRESHOLD))
    prediction_bool = bool(proba >= threshold)

    return PredictionResponse(
        proba_bonne_estimation=proba,
        prediction=prediction_bool,
        seuil_applique=threshold,
        fund_code=features.fund_code,
        ref_unik_asset=features.ref_unik_asset,
    )

@app.get("/health", tags=["diagnostic"])
def health_check():
    return {"status": "ok", "message": "L'API PMVL est opérationnelle."}


@app.post("/predict", response_model=PredictionResponse, tags=["prédiction"])
def predict_pmvl(features: PMVLFeatures):
    try:
        # Exécution de la prédiction existante
        result = run_model_prediction(features)
        
        # --- NOUVEAU CODE POUR LE LOGGING DES OUTPUTS ---
        # On ajoute une entrée spécifique pour les outputs dans un fichier séparé
        # ou on l'ajoute au fichier principal. Ici, on crée un log d'inférence complet.
        inference_log = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    "input_features": features.model_dump(mode="json"),
    "output": {
        "proba_bonne_estimation": result.proba_bonne_estimation,
        "prediction": result.prediction,
        "seuil_applique": result.seuil_applique,
    },
}
        with open(LOG_DIR / "inference_results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(inference_log) + "\n")
        # ------------------------------------------------
            
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de la prédiction : {e}",
        )


# =========================
# Interface Gradio /ui
# =========================

def gradio_predict(
    holding_date,
    pmvl_estim,
    quantity,
    purch_val_clean,
    quote,
    vnc_agrege_dirty,
    entite,
    isin,
    orig_name,
    ticker,
    ref_unik_asset,
    fund_code,
    col_3a,
    canton,
    cic,
    groupe,
    ptf_name,
):
    """
    Fonction appelée par l'interface Gradio.
    Construit un PMVLFeatures puis appelle run_model_prediction.
    """
    features = PMVLFeatures(
        holding_date=holding_date,
        pmvl_estim=pmvl_estim,
        quantity=quantity,
        purch_val_clean=purch_val_clean,
        quote=quote,
        vnc_agrege_dirty=vnc_agrege_dirty,
        entite=entite,
        isin=isin,
        orig_name=orig_name,
        ticker=ticker,
        ref_unik_asset=ref_unik_asset,
        fund_code=fund_code,
        col_3a=col_3a,
        canton=canton,
        cic=cic,
        groupe=groupe,
        ptf_name=ptf_name,
    )

    resp = run_model_prediction(features)
    label = "Estimation jugée BONNE ✅" if resp.prediction else "Estimation jugée MAUVAISE ⚠️"

    return (
        f"Probabilité que la PMVL soit une bonne estimation : {resp.proba_bonne_estimation:.2%}\n"
        f"Seuil appliqué : {resp.seuil_applique:.2f}\n\n"
        f"Décision : {label}\n"
        f"Fonds : {resp.fund_code} | Actif : {resp.ref_unik_asset}"
    )


demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Textbox(label="Date de la position (AAAA-MM-JJ)", value="2026-03-01"),
        gr.Number(label="PMVL estimée", value=1500.0),
        gr.Number(label="Quantité", value=100.0),
        gr.Number(label="Valeur d'achat clean (ptf)", value=45000.0),
        gr.Number(label="Cotation", value=510.0),
        gr.Number(label="VNC agrégée dirty (ptf)", value=49000.0),
        gr.Textbox(label="Entité", value="ENTITE_TEST"),
        gr.Textbox(label="ISIN", value="FR0000000001"),
        gr.Textbox(label="Nom original de l'actif", value="Asset Name Test"),
        gr.Textbox(label="Ticker indice", value="TICKER_TEST"),
        gr.Textbox(label="Réf. unique asset", value="REF_12345"),
        gr.Textbox(label="Code fonds", value="FUND_001"),
        gr.Textbox(label="3A", value="3A_TEST"),
        gr.Textbox(label="Canton", value="CANTON_TEST"),
        gr.Textbox(label="CIC", value="CIC_TEST"),
        gr.Textbox(label="Groupe", value="GROUPE_TEST"),
        gr.Textbox(label="Nom du portefeuille", value="PTF_TEST"),
    ],
    outputs=gr.Textbox(label="Résultat du modèle"),
    title="Scoreur de qualité PMVL",
    description="Interface simple pour évaluer si une estimation PMVL est jugée fiable par le modèle.",
)

# Montage de Gradio sur l'app FastAPI à la racine /
app = gr.mount_gradio_app(app, demo, path="/")