# 🚀 PMVL Quality Prediction API

Ce projet met en place une API de Machine Learning permettant de prédire en temps réel la qualité des estimations de Plus ou Moins-Values Latentes (PMVL). L'API est développée avec **FastAPI**, containerisée avec **Docker**, et déployée de manière automatisée via **CI/CD** sur Hugging Face Spaces.

## 📋 Table des matières
1. [Présentation du projet](#-présentation-du-projet)
2. [Architecture et Outils](#-architecture-et-outils)
3. [Installation et Lancement local](#-installation-et-lancement-local)
4. [Guide d'utilisation de l'API](#-guide-dutilisation-de-lapi)
5. [Monitoring et Logs (Étape 3)](#-monitoring-et-logs-en-production)
6. [Profilage et Optimisations (Étape 4)](#-profilage-et-optimisations)

---

## 🎯 Présentation du projet
L'objectif est d'identifier de manière proactive si l'estimation d'une PMVL est fiable (écart < 5% par rapport à la valeur réelle future). Le modèle sous-jacent est un **CatBoostClassifier** entraîné et géré via MLflow. Un seuil métier (FN vs FP) a été calculé pour optimiser la décision binaire finale.

---

## ⚙️ Architecture et Outils
*   **Modélisation** : Python, Pandas, Scikit-Learn, CatBoost
*   **Tracking** : MLflow
*   **API** : FastAPI, Pydantic (validation des données)
*   **Interface UI** : Gradio
*   **Déploiement** : Docker, GitHub Actions (CI/CD), Hugging Face Spaces
*   **Monitoring MLOps** : Evidently AI (Data Drift), JSONL Logs

---

## 💻 Installation et Lancement local

### 1. Cloner le dépôt
```bash
git clone https://github.com/JoseBravo26/PMVL_assert.git
cd PMVL_assert
```

### 2. Créer un environnement virtuel et installer les dépendances
```bash
python -m venv .venv
source .venv/Scripts/activate  # Sur Windows
# source .venv/bin/activate    # Sur Mac/Linux

pip install -r requirements.txt
```

### 3. Lancer l'API en local
Lancez le serveur uvicorn en pointant vers le fichier `main.py` (qui se trouve dans le dossier `app`).
```bash
uvicorn app.main:app --reload
```
L'API sera disponible sur : `http://127.0.0.1:8000`

---

## 📖 Guide d'utilisation de l'API

L'application expose plusieurs interfaces :

1. **Interface Utilisateur Graphique (Gradio)** :
   *   URL : `/` (Racine de l'application)
   *   Permet de tester manuellement des prédictions via un formulaire visuel simple.

2. **Documentation interactive de l'API (Swagger UI)** :
   *   URL : `/docs`
   *   Permet aux développeurs de tester les endpoints (notamment `/predict`) et de voir le schéma JSON attendu.

3. **Endpoint de Prédiction (`POST /predict`)** :
   Accepte un objet JSON avec les caractéristiques de l'actif et renvoie la probabilité et la décision.

---

## 📊 Monitoring et Logs en production

L'API intègre un système robuste de logging pour suivre les performances MLOps et auditer le **Data Drift**. Chaque requête déclenche l'écriture dans deux fichiers séparés :

*   **`production_logs.jsonl`** : Enregistre les métriques opérationnelles (Latence, Endpoint, Code HTTP, Erreurs).
*   **`inference_results.jsonl`** : Enregistre les features en entrée et les probabilités en sortie pour calculer le Drift avec *Evidently AI*.

### Télécharger les logs depuis l'environnement Cloud
Sur l'environnement de production Hugging Face, vous pouvez télécharger directement les fichiers de logs via ces endpoints dédiés :
*   📥 **Logs opérationnels** : [https://josibra-pmvl-prediction-api.hf.space/download-prod-logs](https://josibra-pmvl-prediction-api.hf.space/download-prod-logs)
*   📥 **Logs d'inférence (Inputs/Outputs)** : [https://josibra-pmvl-prediction-api.hf.space/download-logs](https://josibra-pmvl-prediction-api.hf.space/download-logs)

Un script local `drift_analysis.ipynb` (utilisant *Evidently*) permet ensuite d'analyser ces logs pour générer un rapport de dérive de données.

---

## ⚡ Profilage et Optimisations

Une fois l'API déployée, une analyse de performance a été réalisée à l'aide de `cProfile` pour identifier les goulots d'étranglement lors de l'inférence.

**Goulot identifié :** La construction, la copie et la modification d'objets `pandas.DataFrame` pour préparer une seule ligne de features étaient anormalement coûteuses (overhead).

**Stratégie d'optimisation :** 
La fonction `run_model_prediction` a été réécrite pour se passer totalement de Pandas. Le traitement des valeurs manquantes et le formatage des variables catégorielles sont désormais gérés via des structures natives Python (`list`, `dict`), qui sont envoyées directement au modèle CatBoost.

**Résultats de l'optimisation :**
*   **Latence Standard (Pandas)** : ~9.06 ms
*   **Latence Optimisée (Python Natif)** : ~0.59 ms
*   **Gain de performance** : **93.4 %** d'amélioration du temps d'inférence strict.
*   **Fiabilité** : Aucune régression (différence de probabilité absolue de 0.0).
