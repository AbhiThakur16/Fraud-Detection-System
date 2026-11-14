from pathlib import Path
import os

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_db_connection():
    """Connect to PostgreSQL database."""

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


# ==========================================
# MONITORING SUMMARY
# ==========================================

def get_monitoring_summary():
    """Get overall prediction monitoring statistics."""

    query = """
        SELECT
            COUNT(*) AS total_predictions,

            COUNT(*) FILTER (
                WHERE prediction = 1
            ) AS fraud_alerts,

            COUNT(*) FILTER (
                WHERE prediction = 0
            ) AS normal_transactions,

            ROUND(AVG(fraud_probability), 2)
                AS average_fraud_probability,

            COUNT(*) FILTER (
                WHERE fraud_probability >= 70
            ) AS high_risk_transactions,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE prediction = 1
                ) / NULLIF(COUNT(*), 0),
                2
            ) AS fraud_rate

        FROM prediction_history;
    """

    with get_db_connection() as connection:

        with connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:

            cursor.execute(query)
            result = cursor.fetchone()

    return dict(result)


# ==========================================
# HOURLY PREDICTION TRENDS
# ==========================================

def get_hourly_trends():
    """Get prediction activity during the last 24 hours."""

    query = """
        SELECT
            DATE_TRUNC('hour', created_at)
                AS prediction_hour,

            COUNT(*) AS total_predictions,

            COUNT(*) FILTER (
                WHERE prediction = 1
            ) AS fraud_alerts

        FROM prediction_history

        WHERE created_at >= NOW() - INTERVAL '24 hours'

        GROUP BY DATE_TRUNC('hour', created_at)

        ORDER BY prediction_hour;
    """

    with get_db_connection() as connection:

        with connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:

            cursor.execute(query)
            results = cursor.fetchall()

    return [dict(row) for row in results]


# ==========================================
# HIGH-RISK TRANSACTIONS
# ==========================================

def get_high_risk_predictions(limit=20):
    """Get transactions having fraud probability >= 70%."""

    query = """
        SELECT
            transaction_id,
            transaction_amount,
            fraud_probability,
            prediction,
            status,
            created_at

        FROM prediction_history

        WHERE fraud_probability >= 70

        ORDER BY fraud_probability DESC

        LIMIT %s;
    """

    with get_db_connection() as connection:

        with connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:

            cursor.execute(query, (limit,))
            results = cursor.fetchall()

    return [dict(row) for row in results]


# ==========================================
# PREDICTION CONFIDENCE MONITORING
# ==========================================

def get_prediction_confidence():
    """
    Analyze fraud probability distribution
    from stored predictions.
    """

    query = """
        SELECT
            COUNT(*) AS total_predictions,

            ROUND(AVG(fraud_probability), 2)
                AS average_probability,

            ROUND(MIN(fraud_probability), 2)
                AS minimum_probability,

            ROUND(MAX(fraud_probability), 2)
                AS maximum_probability,

            COUNT(*) FILTER (
                WHERE fraud_probability < 20
            ) AS very_low_risk,

            COUNT(*) FILTER (
                WHERE fraud_probability >= 20
                AND fraud_probability < 40
            ) AS low_risk,

            COUNT(*) FILTER (
                WHERE fraud_probability >= 40
                AND fraud_probability < 60
            ) AS medium_risk,

            COUNT(*) FILTER (
                WHERE fraud_probability >= 60
                AND fraud_probability < 80
            ) AS high_risk,

            COUNT(*) FILTER (
                WHERE fraud_probability >= 80
            ) AS very_high_risk

        FROM prediction_history;
    """

    with get_db_connection() as connection:

        with connection.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:

            cursor.execute(query)
            result = cursor.fetchone()

    return dict(result)