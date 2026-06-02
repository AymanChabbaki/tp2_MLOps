import pandas as pd
import pytest
from src.validate import validate_data

def test_validate_data_success():
    # Dataset valide
    data = {
        "client_id": [1, 2, 3],
        "age": [25, 45, 65],
        "income": [50000.0, 60000.0, 75000.0],
        "loan_amount": [15000.0, 20000.0, 30000.0],
        "months_employed": [24, 60, 120],
        "default": [0, 1, 0]
    }
    df = pd.DataFrame(data)
    assert validate_data(df) is True

def test_validate_data_missing_column():
    # Dataset sans la colonne 'default'
    data = {
        "client_id": [1, 2],
        "age": [25, 45],
        "income": [50000.0, 60000.0],
        "loan_amount": [15000.0, 20000.0],
        "months_employed": [24, 60]
    }
    df = pd.DataFrame(data)
    assert validate_data(df) is False

def test_validate_data_invalid_target():
    # Valeurs hors de {0, 1} dans 'default'
    data = {
        "client_id": [1, 2],
        "age": [25, 45],
        "income": [50000.0, 60000.0],
        "loan_amount": [15000.0, 20000.0],
        "months_employed": [24, 60],
        "default": [0, 99]
    }
    df = pd.DataFrame(data)
    assert validate_data(df) is False
