
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def get_connection():
    """
    Establish a PostgreSQL database connection.

    Supports:
    - Local PostgreSQL
    - Render PostgreSQL
    - SSL-secured cloud connections
    """

    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode=os.getenv("DB_SSLMODE", "require")
        )

        return connection

    except psycopg2.Error as error:
        print(f"Database connection failed: {error}")
        raise


if __name__ == "__main__":

    connection = None

    try:
        connection = get_connection()

        print("Database connection successful!")

        cursor = connection.cursor()

        cursor.execute("SELECT version();")

        database_version = cursor.fetchone()

        print("PostgreSQL Version:")
        print(database_version[0])

        cursor.execute("SELECT current_database();")

        database_name = cursor.fetchone()

        print("Connected Database:")
        print(database_name[0])

        cursor.close()

    except Exception as error:
        print(f"Connection test failed: {error}")

    finally:
        if connection:
            connection.close()
            print("Database connection closed.")
