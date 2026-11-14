
import json
from pathlib import Path

import pandas as pd
from psycopg2.extras import RealDictCursor

from src.prediction_repository import get_db_connection


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

REFERENCE_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "fraud_features_v2.csv"
)


# ==========================================
# LOAD REFERENCE DATA
# ==========================================

def get_reference_data():

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {REFERENCE_DATA_PATH}"
        )

    return pd.read_csv(REFERENCE_DATA_PATH)


# ==========================================
# LOAD CURRENT LIVE DATA
# ==========================================

def get_current_data(limit=1000):

    query = """
        SELECT input_features
        FROM prediction_history
        WHERE input_features IS NOT NULL
        ORDER BY created_at DESC
        LIMIT %s;
    """

    connection = get_db_connection()

    try:

        with connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:

            cursor.execute(query, (limit,))
            records = cursor.fetchall()

        transactions = []

        for record in records:

            features = record["input_features"]

            if isinstance(features, str):
                features = json.loads(features)

            if isinstance(features, dict):
                transactions.append(features)

        return pd.DataFrame(transactions)

    finally:
        connection.close()


# ==========================================
# LOAD BOTH DATASETS
# ==========================================

def get_drift_datasets(limit=1000):

    reference_df = get_reference_data()

    current_df = get_current_data(limit)

    return reference_df, current_df
