def predict(model, X, model_type="multiclass"):
    """
    Prédit les résultats selon le type de modèle.
    Args:
        model: modèle entraîné (sklearn, keras, YOLO, etc.)
        X: données d'entrée (array, image, etc.)
        model_type: "multiclass", "oneclass" ou "yolo"
    Returns:
        prédictions du modèle
    """
    if model_type == "multiclass":
        return model.predict(X)
    elif model_type == "oneclass":
        return model.predict(X)
    elif model_type == "yolo":
        # Supposons que X est une image ou un chemin d'image
        results = model(X)
        return results
    else:
        raise ValueError("Type de modèle non supporté")
