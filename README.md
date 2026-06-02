# Pipeline de Machine Learning de Score de Risque Client

Ce dépôt implémente une architecture industrialisée, reproductible et automatisée pour l'entraînement, la validation et l'évaluation d'un modèle de classification du score de risque client.

## 🏗️ Architecture du Projet

```
├── data/
│   ├── raw/           # Données brutes de départ (gérées par DVC)
│   └── processed/     # Ensembles Train/Test prétraités (gérés par DVC)
├── src/
│   ├── ingest.py      # Chargement et génération de données clients synthétiques
│   ├── validate.py    # Contrôle de la conformité du schéma et anomalies de données
│   ├── transform.py   # Nettoyage, feature engineering et split train/test
│   ├── train.py       # Entraînement Random Forest et tracking MLflow
│   └── evaluate.py    # Calcul des métriques de test et Gate pour le Model Registry
├── configs/
│   └── params.yaml    # Chemins de données, hyperparamètres et seuils
├── tests/
│   └── test_data.py   # Tests unitaires Pytest
├── Dockerfile         # Environnement de conteneurisation de production
├── requirements.txt   # Dépendances Python
├── dvc.yaml           # Pipeline DAG et dépendances reproductibles
├── .gitignore         # Exclusion des fichiers locaux et de données de Git
└── README.md          # Documentation utilisateur
```

## 🛠️ Stack Technique

- **Python 3.10+**
- **DVC** : Orchestration et mise en cache des étapes du pipeline.
- **MLflow** : Suivi des expériences et enregistrement des modèles candidats.
- **Docker** : Empaquetage du pipeline.
- **Scikit-Learn & Pandas** : Modélisation et traitement.
- **Pytest** : Validation de la structure de données.

---

## 🚀 Installation & Exécution locale

### 1. Installation de l'environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Lancement de MLflow Tracking local

Dans un terminal séparé, lancez le serveur MLflow local pour le suivi :
```bash
mlflow ui
```
Par défaut, le serveur écoute sur [http://localhost:5000](http://localhost:5000).

### 3. Exécution avec DVC (DAG)

Initialisez DVC et lancez le pipeline :
```bash
dvc init --no-scm  # (Ou simplement git init puis dvc init)
dvc repro
```
DVC va automatiquement exécuter les étapes séquentiellement :
1. `ingest`
2. `validate`
3. `transform`
4. `train`
5. `evaluate`

En cas de relancement sans modification de code ou de paramètres, DVC utilisera son cache.

### 4. Lancement des tests unitaires

```bash
pytest tests/
```

---

## 🐳 Conteneurisation (Docker)

Pour exécuter le pipeline de façon isolée sous Docker :

```bash
# Build de l'image
docker build -t risk-scoring-pipeline .

# Exécution du pipeline
docker run --network="host" risk-scoring-pipeline
```
*(Le flag `--network="host"` permet au conteneur de communiquer directement avec votre instance MLflow locale sur `localhost:5000`)*.
