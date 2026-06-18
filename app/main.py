from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import os
import pandas as pd
import numpy as np
import gradio as gr
import time
from .logging_utils import log_prediction_event
from .schemas import PMVLFeatures, PredictionResponse
from .model_loader import get_model, get_feature_columns

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
    Inclut la génération de logs de production (inputs, outputs, latence).
    """
    start_time = time.perf_counter()
    status = "success"
    error_message = None

    try:
        model = get_model()
        feature_columns = get_feature_columns()

        # 1) Récupérer les données brutes
        raw_dict = features.model_dump(by_alias=True)
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

        response = PredictionResponse(
            proba_bonne_estimation=proba,
            prediction=prediction_bool,
            seuil_applique=threshold,
            fund_code=features.fund_code,
            ref_unik_asset=features.ref_unik_asset,
        )
        return response

    except Exception as e:
        status = "error"
        error_message = str(e)
        raise

    finally:
        # 7) Logging structuré dans tous les cas (succès ou erreur)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Formatage des sorties pour le log (mode='json' convertit les objets complexes)
        outputs_log = response.model_dump(mode='json') if status == "success" else {"error": error_message}
        
        log_prediction_event(
            inputs=features.model_dump(by_alias=True, mode='json'),
            outputs=outputs_log,
            status=status,
            latency_ms=latency_ms,
        )

@app.get("/health", tags=["diagnostic"])
def health_check():
    return {"status": "ok", "message": "L'API PMVL est opérationnelle."}


@app.post("/predict", response_model=PredictionResponse, tags=["prédiction"])
def predict_pmvl(features: PMVLFeatures):
    try:
        return run_model_prediction(features)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de la prédiction : {e}",
        )

from fastapi.responses import FileResponse

@app.get("/download-logs", tags=["diagnostic"])
def download_logs():
    """
    Route temporaire pour télécharger les logs générés en production sur Hugging Face.
    """
    log_file_path = Path("logs/inference_results.jsonl")
    
    if log_file_path.exists():
        return FileResponse(
            path=log_file_path, 
            filename="inference_results_HF.jsonl",
            media_type="application/json"
        )
    else:
        return {"error": "Le fichier de logs n'existe pas encore. Faites des prédictions d'abord."}

@app.get("/download-prod-logs", tags=["diagnostic"])
def download_prod_logs():
    """
    Route temporaire pour télécharger les logs du middleware.
    """
    log_file_path = Path("logs/production_logs.jsonl")
    
    if log_file_path.exists():
        return FileResponse(
            path=log_file_path, 
            filename="production_logs_HF.jsonl",
            media_type="application/json"
        )
    else:
        return {"error": "Le fichier de logs de production n'existe pas encore."}

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