FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY dvc.yaml .
COPY README.md .

# Configurer l'URI MLflow pour pointer vers l'hôte si besoin
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Par défaut, on lance le pipeline complet avec DVC
CMD ["dvc", "repro"]
