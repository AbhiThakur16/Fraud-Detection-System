import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Connect Python application to PostgreSQL."""

    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )

        return connection

    except psycopg2.Error as error:
        print(f"Database connection failed: {error}")
        raise


if __name__ == "__main__":

    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                EXTRACT(HOUR FROM occurred_at)::INTEGER
                    AS transaction_hour,

                COUNT(*) AS total_transactions,

                COUNT(*) FILTER (
                    WHERE is_fraud = TRUE
                ) AS fraud_count,

                ROUND(
                    COUNT(*) FILTER (
                        WHERE is_fraud = TRUE
                    ) * 100.0 / COUNT(*),
                    2
                ) AS fraud_percentage

            FROM transactions

            GROUP BY transaction_hour

            ORDER BY transaction_hour;
        """)

        print("\n===================================")
        print("HOUR-WISE FRAUD ANALYSIS")
        print("===================================")

        for row in cursor.fetchall():

            print(
                f"Hour: {row[0]:02d}:00 | "
                f"Total: {row[1]} | "
                f"Fraud: {row[2]} | "
                f"Fraud Rate: {row[3]}%"
            )

        cursor.execute("""
            SELECT
                EXTRACT(DOW FROM occurred_at)::INTEGER
                    AS transaction_day,

                TO_CHAR(occurred_at, 'FMDay')
                    AS day_name,

                COUNT(*) AS total_transactions,

                COUNT(*) FILTER (
                    WHERE is_fraud = TRUE
                ) AS fraud_count

            FROM transactions

            GROUP BY transaction_day, day_name

            ORDER BY transaction_day;
        """)

        print("\n===================================")
        print("DAY-WISE FRAUD ANALYSIS")
        print("===================================")

        for row in cursor.fetchall():

            print(
                f"Day: {row[1].strip()} | "
                f"Total: {row[2]} | "
                f"Fraud: {row[3]}"
            )

        print("\nAnalysis completed successfully.")

    except Exception as error:
        print(f"Error: {error}")

    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()

        print("\nDatabase connection closed.")