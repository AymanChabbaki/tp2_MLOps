import os
import argparse
import logging
import yaml
import numpy as np
import pandas as pd

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def generate_synthetic_data(num_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Génère un jeu de données synthétique réaliste pour le scoring de crédit."""
    np.random.seed(random_state)
    logger.info(f"Génération de {num_samples} échantillons de données clients...")
    
    # Caractéristiques des clients
    age = np.random.randint(18, 70, size=num_samples)
    income = np.random.normal(45000, 15000, size=num_samples).clip(10000, 150000)
    loan_amount = income * np.random.uniform(0.1, 0.6, size=num_samples)
    months_employed = np.random.randint(0, 120, size=num_samples)
    
    # Calcul d'une probabilité de défaut (cible) corrélée avec le revenu, le montant du prêt et la durée de l'emploi
    score = (
        0.5 * (loan_amount / income) 
        - 0.3 * (income / 100000) 
        - 0.2 * (months_employed / 120) 
        + 0.1 * (age / 70) 
        + np.random.normal(0, 0.1, size=num_samples)
    )
    # Convertir en binaire (0 ou 1)
    default = (score > np.percentile(score, 80)).astype(int)
    
    df = pd.DataFrame({
        "client_id": np.arange(1000, 1000 + num_samples),
        "age": age,
        "income": income.round(2),
        "loan_amount": loan_amount.round(2),
        "months_employed": months_employed,
        "default": default
    })
    
    # Ajouter quelques valeurs manquantes pour la robustesse (ex: dans income)
    mask = np.random.choice([True, False], size=num_samples, p=[0.02, 0.98])
    df.loc[mask, "income"] = np.nan
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Étape 1 : Chargement / Ingestion des données")
    parser.add_argument("--config", type=str, default="configs/params.yaml", help="Chemin vers le fichier de config params.yaml")
    parser.add_argument("--output", type=str, help="Chemin de sortie du fichier brut")
    args = parser.parse_args()
    
    try:
        logger.info(f"Lecture de la configuration depuis {args.config}")
        with open(args.config, "r") as f:
            params = yaml.safe_load(f)
            
        output_path = args.output if args.output else params["data"]["raw_path"]
        random_state = params["data"]["random_state"]
        
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Générer et stocker le dataset
        df = generate_synthetic_data(num_samples=1200, random_state=random_state)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Données brutes sauvegardées avec succès dans {output_path} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    except Exception as e:
        logger.error(f"Échec de l'étape d'ingestion : {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
