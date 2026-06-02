import os
import argparse
import logging
import pickle
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def train_model(train_df: pd.DataFrame, model_params: dict):
    """Entraîne un modèle Random Forest Classifier."""
    X_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]
    
    logger.info("Instanciation et entraînement du modèle Random Forest...")
    model = RandomForestClassifier(
        n_estimators=model_params.get("n_estimators", 100),
        max_depth=model_params.get("max_depth", 6),
        class_weight=model_params.get("class_weight", "balanced"),
        random_state=model_params.get("random_state", 42)
    )
    model.fit(X_train, y_train)
    
    # Évaluation rapide sur le train set
    preds = model.predict(X_train)
    probs = model.predict_proba(X_train)[:, 1]
    
    metrics = {
        "train_accuracy": accuracy_score(y_train, preds),
        "train_precision": precision_score(y_train, preds, zero_division=0),
        "train_recall": recall_score(y_train, preds, zero_division=0),
        "train_auc": roc_auc_score(y_train, probs)
    }
    
    logger.info(f"Performances d'entraînement : AUC={metrics['train_auc']:.4f}, Accuracy={metrics['train_accuracy']:.4f}")
    return model, metrics

def main():
    parser = argparse.ArgumentParser(description="Étape 4 : Entraînement du modèle avec MLflow")
    parser.add_argument("--config", type=str, default="configs/params.yaml", help="Chemin vers params.yaml")
    parser.add_argument("--train-in", type=str, help="Chemin vers les données d'entraînement")
    parser.add_argument("--model-out", type=str, help="Chemin de sortie du modèle sauvegardé (pickle)")
    args = parser.parse_args()
    
    try:
        logger.info(f"Lecture de la configuration depuis {args.config}")
        with open(args.config, "r") as f:
            params = yaml.safe_load(f)
            
        train_path = args.train_in if args.train_in else params["data"]["train_path"]
        model_out_path = args.model_out if args.model_out else os.path.join(params["data"]["processed_dir"], "model.pkl")
        
        # Configuration MLflow
        experiment_name = params["mlflow"]["experiment_name"]
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"Chargement des données d'entraînement depuis {train_path}")
        train_df = pd.read_csv(train_path)
        
        # Début du run MLflow
        with mlflow.start_run(run_name="Training_Stage") as run:
            logger.info(f"Début du run MLflow (ID: {run.info.run_id})")
            
            # Log des paramètres
            mlflow.log_params(params["model"])
            
            # Entraînement
            model, train_metrics = train_model(train_df, params["model"])
            
            # Log des métriques d'entraînement
            mlflow.log_metrics(train_metrics)
            
            # Sauvegarde locale du modèle sous format pickle pour le stage suivant
            os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
            with open(model_out_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Modèle sérialisé localement dans {model_out_path}")
            
            # Log du modèle dans MLflow (sans enregistrement immédiat, géré par le stage d'évaluation)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=train_df.drop(columns=["default"]).head(5)
            )
            
            # Enregistrement de l'ID du run MLflow dans un fichier temporaire pour le stage d'évaluation
            run_id_path = model_out_path.replace("model.pkl", "mlflow_run_id.txt")
            with open(run_id_path, "w") as f:
                f.write(run.info.run_id)
            logger.info(f"Run ID MLflow ({run.info.run_id}) écrit dans {run_id_path}")
            
    except Exception as e:
        logger.error(f"Échec de l'étape d'entraînement : {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
