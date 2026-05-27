
# Logistic Regression Production Predictor
def predict_pmvl_failure(model, X, threshold=0.4):
    """PMVL failure predictor (production)\n
    Returns: 1=failure predicted, 0=success predicted
    """
    probs = model.predict_proba(X)[:, 1]
    return (probs >= threshold).astype(int)
