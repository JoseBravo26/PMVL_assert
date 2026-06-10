FROM python:3.12-slim

# 1. Création d'un utilisateur standard (UID 1000 requis par Hugging Face)
RUN useradd -m -u 1000 user
USER user

# 2. Variables d'environnement pour l'utilisateur et Python
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Répertoire de travail
WORKDIR $HOME/app

# 4. Copie des dépendances avec les bons droits
COPY --chown=user requirements.txt .

# 5. Mise à jour de pip (pour enlever le 'notice') et installation
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copie du code et des modèles
COPY --chown=user app/ ./app/
COPY --chown=user models/ ./models/

# 7. Variables de l'API avec le nouveau chemin
ENV GLOBAL_THRESHOLD=0.45
ENV MODEL_PATH=$HOME/app/models/pmvl_catboost_final.cbm
ENV FEATURES_PATH=$HOME/app/models/pmvl_feature_columns.txt

# 8. Exposer le port 7860 (Port par défaut de Hugging Face Spaces)
EXPOSE 7860

# 9. Création du dossier de logs avec les permissions nécessaires
RUN mkdir -p $HOME/app/logs && chmod -R 777 $HOME/app/logs

# 10. Démarrage de l'API sur le port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]