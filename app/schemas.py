from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import date

class PMVLFeatures(BaseModel):
    """
    Schéma Pydantic représentant une observation d'entrée pour la prédiction PMVL.
    Les noms des attributs sont des alias Python-friendly pour les colonnes complexes du DataFrame.
    """
    # ------------------ Colonnes de type Date ------------------
    holding_date: date = Field(
        ..., 
        alias="PMVL[Holding date]",
        description="Date de la position (Trading/Holding date)"
    )
    quote_date: Optional[str] = Field(
        None, 
        alias="PMVL[Quote Date]",
        description="Date de la cotation (peut contenir des valeurs manquantes)"
    )

    # ------------------ Colonnes Numériques (float64) ------------------
    pmvl_estim: Optional[float] = Field(
        None, 
        alias="PMVL[PMVL Estimé]",
        description="Estimation de la PMVL au temps t"
    )
    pfm_indice_perf: Optional[float] = Field(
        None, 
        alias="PMVL[PFMIndice.Perf J / J-1]",
        description="Performance de l'indice J / J-1"
    )
    prmp_pmvl: Optional[float] = Field(
        None, 
        alias="PMVL[PRMP PMVL]",
        description="Realized PMVL at t"
    )
    prmp_vnc: Optional[float] = Field(
        None, 
        alias="PMVL[PRMP VNC]",
        description="Realized VNC at t"
    )
    prmp_mtm: Optional[float] = Field(
        None, 
        alias="PMVL[PRMP MtM]",
        description="Realized MtM at t"
    )
    quantity: float = Field(
        ..., 
        alias="PMVL[Quantity]",
        description="Quantité de l'actif"
    )
    purch_val_clean: float = Field(
        ..., 
        alias="PMVL[Purch. Val. (clean) (ptf cur.)]",
        description="Valeur d'achat clean en devise du portefeuille"
    )
    quote: float = Field(
        ..., 
        alias="PMVL[Quote]",
        description="Cotation de l'actif"
    )
    vnc_agrege_dirty: float = Field(
        ..., 
        alias="PMVL[VNC Agrege dirty (ptf cur.)]",
        description="VNC Agrégée dirty en devise du portefeuille"
    )

    # ------------------ Colonnes Catégorielles (object) ------------------
    entite: str = Field(
        ..., 
        alias="PMVL[ENTITE]",
        description="Entité légale"
    )
    isin: str = Field(
        ..., 
        alias="PMVL[ISIN]",
        description="Code ISIN de l'instrument"
    )
    orig_name: str = Field(
        ..., 
        alias="PMVL[Orig. name]",
        description="Nom original de l'actif"
    )
    ticker: str = Field(
        ..., 
        alias="PMVL[Parametres_Indices.TICKER]",
        description="Ticker de l'indice"
    )
    ref_unik_asset: str = Field(
        ..., 
        alias="PMVL[Ref Unik Asset]",
        description="Référence unique de l'asset"
    )
    fund_code: str = Field(
        ..., 
        alias="PMVL[Selected Fund code]",
        description="Code du fonds sélectionné"
    )
    col_3a: str = Field(
        ..., 
        alias="PMVL[3A]",
        description="Classification 3A"
    )
    canton: str = Field(
        ..., 
        alias="PMVL[CANTON]",
        description="Canton"
    )
    cic: str = Field(
        ..., 
        alias="PMVL[CIC]",
        description="Classification CIC"
    )
    groupe: str = Field(
        ..., 
        alias="PMVL[GROUPE]",
        description="Groupe"
    )
    ptf_name: str = Field(
        ..., 
        alias="PMVL[Ptf name]",
        description="Nom du portefeuille"
    )

    model_config = ConfigDict(populate_by_name=True)


class PredictionResponse(BaseModel):
    """
    Schéma de la réponse retournée par l'API.
    """
    proba_bonne_estimation: float = Field(
        ...,
        description="Probabilité (entre 0 et 1) que la PMVL soit une bonne estimation."
    )
    prediction: bool = Field(
        ...,
        description="True si la PMVL est jugée bonne (proba >= seuil), False sinon."
    )
    seuil_applique: float = Field(
        ...,
        description="Le seuil de probabilité utilisé pour cette décision (ex: 0.45)."
    )
    fund_code: str = Field(
        ...,
        description="Rappel du code du fonds pour traçabilité."
    )
    ref_unik_asset: str = Field(
        ...,
        description="Rappel de la référence de l'actif pour traçabilité."
    )