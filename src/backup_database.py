import os
import pandas as pd

from sqlalchemy import create_engine, URL
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def get_database_engine():

    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME")
    )

    return create_engine(connection_url)


def backup_database():

    print("\n===================================")
    print("DATABASE BACKUP")
    print("===================================")

    backup_directory = "data/backup"

    os.makedirs(
        backup_directory,
        exist_ok=True
    )

    tables = [
        "users",
        "accounts",
        "merchants",
        "transactions",
        "fraud_alerts",
        "transaction_audit_logs"
    ]

    engine = get_database_engine()

    try:

        for table in tables:

            print(f"\nBacking up table: {table}")

            query = f'SELECT * FROM "{table}"'

            df = pd.read_sql_query(
                query,
                engine
            )

            output_path = os.path.join(
                backup_directory,
                f"{table}_backup.csv"
            )

            df.to_csv(
                output_path,
                index=False
            )

            print(f"Records: {len(df)}")

            print(f"Saved: {output_path}")

        print("\n===================================")
        print("DATABASE BACKUP COMPLETED")
        print("===================================")

    finally:

        engine.dispose()


if __name__ == "__main__":

    backup_database()