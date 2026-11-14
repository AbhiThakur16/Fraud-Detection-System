
import os
import json
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_db_connection():
    """Connect to PostgreSQL."""

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def save_prediction(record):
    """Save prediction and original transaction features into PostgreSQL."""

    query = """
        INSERT INTO prediction_history (
            transaction_id,
            transaction_amount,
            fraud_probability,
            prediction,
            status,
            threshold,
            input_features
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
        RETURNING id;
    """

    input_features = record.get("input_features")

    if input_features is not None:
        input_features = json.dumps(input_features, default=str)

    values = (
        record["transaction_id"],
        record["transaction_amount"],
        record["fraud_probability"],
        record["prediction"],
        record["status"],
        record["threshold"],
        input_features
    )

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, values)
            record_id = cursor.fetchone()[0]

        connection.commit()
        return record_id

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_prediction_history(limit=100):
    """Fetch recent prediction history."""

    query = """
        SELECT
            transaction_id,
            transaction_amount,
            fraud_probability,
            prediction,
            status,
            threshold,
            created_at
        FROM prediction_history
        ORDER BY created_at DESC
        LIMIT %s;
    """

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()

        columns = [
            "transaction_id",
            "transaction_amount",
            "fraud_probability",
            "prediction",
            "status",
            "threshold",
            "timestamp"
        ]

        return [
            dict(zip(columns, row))
            for row in rows
        ]

    finally:
        connection.close()


def get_prediction_summary():
    """Fetch fraud prediction statistics."""

    query = """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (WHERE prediction = 1),
            COUNT(*) FILTER (WHERE prediction = 0)
        FROM prediction_history;
    """

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            row = cursor.fetchone()

        return {
            "total_predictions": row[0],
            "fraud_alerts": row[1],
            "normal_transactions": row[2]
        }

    finally:
        connection.close()
