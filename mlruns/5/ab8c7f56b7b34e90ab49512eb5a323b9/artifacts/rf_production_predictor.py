
# RF v2 Production Predictor (F1 class 0: 0.562)
def predict_pmvl_failure(model, X, threshold=0.9):
    """Predict PMVL failures >=5% tomorrow (production ready)\n
    Returns: 1=failure predicted, 0=success predicted
    """
    probs = model.predict_proba(X)[:, 1]
    return (probs >= threshold).astype(int)
