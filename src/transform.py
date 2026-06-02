import os
import argparse
import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def preprocess_and_split(df: pd.DataFrame, split_ratio: float, random_state: int):
    """Prétraite les données (imputation) et les sépare en ensembles Train/Test."""
    logger.info("Début du prétraitement des données...")
    
    # 1. Imputation des valeurs manquantes (ex: Médiane pour le revenu)
    if df["income"].isnull().any():
        median_income = df["income"].median()
        logger.info(f"Imputation des valeurs manquantes de 'income' par la médiane : {median_income}")
        df["income"] = df["income"].fillna(median_income)
        
    # 2. Feature Engineering simple (Ratio Prêt/Revenu)
    logger.info("Création de la feature 'loan_to_income_ratio'")
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["income"] + 1e-5)
    
    # 3. Suppression des colonnes inutiles pour l'entraînement
    features_to_drop = ["client_id"]
    df_clean = df.drop(columns=features_to_drop)
    
    # 4. Séparation Train / Test
    logger.info(f"Découpage Train/Test (Ratio test: {split_ratio}, Random State: {random_state})")
    train_df, test_df = train_test_split(
        df_clean, 
        test_size=split_ratio, 
        random_state=random_state, 
        stratify=df_clean["default"]
    )
    
    return train_df, test_df

def main():
    parser = argparse.ArgumentParser(description="Étape 3 : Feature Engineering et Découpage des données")
    parser.add_argument("--config", type=str, default="configs/params.yaml", help="Chemin vers params.yaml")
    parser.add_argument("--input", type=str, help="Chemin d'entrée des données validées")
    parser.add_argument("--train-out", type=str, help="Chemin de sortie du train set")
    parser.add_argument("--test-out", type=str, help="Chemin de sortie du test set")
    args = parser.parse_args()
    
    try:
        logger.info(f"Lecture de la configuration depuis {args.config}")
        with open(args.config, "r") as f:
            params = yaml.safe_load(f)
            
        input_path = args.input if args.input else params["data"]["raw_path"].replace("raw", "processed").replace(".csv", "_validated.csv")
        train_path = args.train_out if args.train_out else params["data"]["train_path"]
        test_path = args.test_out if args.test_out else params["data"]["test_path"]
        
        split_ratio = params["data"]["split_ratio"]
        random_state = params["data"]["random_state"]
        
        logger.info(f"Chargement des données validées depuis {input_path}")
        df = pd.read_csv(input_path)
        
        train_df, test_df = preprocess_and_split(df, split_ratio, random_state)
        
        # Création des dossiers de sortie
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Fichiers d'entraînement et de test créés avec succès : {train_path} ({train_df.shape[0]} lignes), {test_path} ({test_df.shape[0]} lignes)")
        
    except Exception as e:
        logger.error(f"Échec de l'étape de transformation : {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
