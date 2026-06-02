import os
import argparse
import logging
import yaml
import pandas as pd

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "client_id": "int64",
    "age": "int64",
    "income": "float64",
    "loan_amount": "float64",
    "months_employed": "int64",
    "default": "int64"
}

def validate_data(df: pd.DataFrame) -> bool:
    """Valide les données selon les règles métiers et le schéma attendu."""
    is_valid = True
    
    # 1. Vérification des colonnes requises
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.error(f"Colonnes manquantes dans le dataset : {missing_cols}")
        return False
        
    logger.info("Validation des colonnes : Succès")
    
    # 2. Vérification des types de données (vérification souple ou conversion)
    for col, expected_type in REQUIRED_COLUMNS.items():
        if df[col].isnull().any():
            logger.warning(f"La colonne '{col}' contient des valeurs manquantes ({df[col].isnull().sum()} valeurs).")
        # On s'assure que le type de base est compatible (ex: float pour les colonnes numériques)
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f"La colonne '{col}' n'est pas de type numérique.")
            is_valid = False

    # 3. Vérification des plages de valeurs cohérentes
    if (df["age"] < 18).any() or (df["age"] > 120).any():
        logger.warning("Présence d'âges suspects (< 18 ans ou > 120 ans) dans le jeu de données.")
        
    if ((df["default"] != 0) & (df["default"] != 1)).any():
        logger.error("La colonne cible 'default' contient des valeurs autres que 0 et 1.")
        is_valid = False
        
    return is_valid

def main():
    parser = argparse.ArgumentParser(description="Étape 2 : Contrôle du schéma et anomalies")
    parser.add_argument("--config", type=str, default="configs/params.yaml", help="Chemin vers params.yaml")
    parser.add_argument("--input", type=str, help="Chemin d'entrée des données brutes")
    parser.add_argument("--output", type=str, help="Chemin de sortie des données validées")
    args = parser.parse_args()
    
    try:
        logger.info(f"Lecture de la configuration depuis {args.config}")
        with open(args.config, "r") as f:
            params = yaml.safe_load(f)
            
        input_path = args.input if args.input else params["data"]["raw_path"]
        # On définit une sortie validée temporaire ou intermédiaire
        output_path = args.output if args.output else input_path.replace("raw", "processed").replace(".csv", "_validated.csv")
        
        logger.info(f"Chargement des données pour validation depuis {input_path}")
        df = pd.read_csv(input_path)
        
        if not validate_data(df):
            raise ValueError("La validation des données a échoué. Arrêt du pipeline.")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Validation réussie. Données transférées vers {output_path}")
        
    except Exception as e:
        logger.error(f"Échec de l'étape de validation : {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
