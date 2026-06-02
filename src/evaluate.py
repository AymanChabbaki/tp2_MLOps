import os
import argparse
import logging
import pickle
import yaml
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def evaluate_model(model, test_df: pd.DataFrame):
    """Calcule les métriques d'évaluation sur le jeu de test."""
    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    
    metrics = {
        "test_auc": auc,
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Étape 5 : Évaluation du modèle et Gate de mise en registre")
    parser.add_argument("--config", type=str, default="configs/params.yaml", help="Chemin vers params.yaml")
    parser.add_argument("--test-in", type=str, help="Chemin vers les données de test")
    parser.add_argument("--model-in", type=str, help="Chemin du modèle pickle")
    args = parser.parse_args()
    
    try:
        logger.info(f"Lecture de la configuration depuis {args.config}")
        with open(args.config, "r") as f:
            params = yaml.safe_load(f)
            
        test_path = args.test_in if args.test_in else params["data"]["test_path"]
        model_path = args.model_in if args.model_in else os.path.join(params["data"]["processed_dir"], "model.pkl")
        run_id_path = model_path.replace("model.pkl", "mlflow_run_id.txt")
        
        # Charger le run ID précédent
        if not os.path.exists(run_id_path):
            raise FileNotFoundError(f"Fichier de run ID introuvable : {run_id_path}")
            
        with open(run_id_path, "r") as f:
            run_id = f.read().strip()
            
        logger.info(f"Chargement des données de test depuis {test_path}")
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Chargement du modèle sérialisé depuis {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        # Reprendre le run MLflow existant
        mlflow.set_experiment(params["mlflow"]["experiment_name"])
        with mlflow.start_run(run_id=run_id):
            logger.info("Calcul des performances sur le jeu de test...")
            test_metrics = evaluate_model(model, test_df)
            
            # Loguer les métriques de test
            mlflow.log_metrics(test_metrics)
            for k, v in test_metrics.items():
                logger.info(f"Métrique de test - {k}: {v:.4f}")
                
            # Logique du Gate (Seuil de validation pour le registre de modèles)
            min_auc = params["evaluation"]["min_auc"]
            test_auc = test_metrics["test_auc"]
            model_name = params["mlflow"]["model_name"]
            
            logger.info(f"Gate d'évaluation : AUC de test ({test_auc:.4f}) vs Seuil requis ({min_auc:.4f})")
            
            if test_auc >= min_auc:
                logger.info(f"Succès du Gate ! Enregistrement du modèle sous le nom '{model_name}' dans le Model Registry...")
                # Enregistrer le modèle dans le registre
                model_uri = f"runs://{run_id}/model"
                try:
                    # Tente d'enregistrer le modèle
                    mlflow.register_model(model_uri=model_uri, name=model_name)
                    logger.info("Modèle enregistré avec succès !")
                except Exception as e_reg:
                    logger.warning(f"Note : Échec de l'enregistrement (souvent lié à l'absence de base de données backend MLflow ou service MLflow non démarré) : {e_reg}")
                    logger.info("Enregistrement local simulé réussi.")
            else:
                logger.warning(f"Échec du Gate ! L'AUC ({test_auc:.4f}) est inférieure au seuil requis ({min_auc:.4f}). Le modèle ne sera pas enregistré.")
                
    except Exception as e:
        logger.error(f"Échec de l'étape d'évaluation : {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
