import pandas as pd
import numpy as np

from src.db_connection import get_connection


def load_transaction_data():

    print("\nConnecting to PostgreSQL...")

    conn = get_connection()

    query = """
        SELECT
            transaction_id,
            sender_account_id,
            receiver_account_id,
            merchant_id,
            transaction_type,
            transaction_channel,
            amount,
            currency,
            transaction_status,
            is_fraud,
            occurred_at

        FROM transactions;
    """

    try:
        df = pd.read_sql_query(query, conn)

        print("Data loaded successfully!")
        print(f"Total Records: {len(df)}")

        return df

    finally:
        conn.close()


def create_features(df):

    print("\nStarting Feature Engineering...")

    # Convert timestamp
    df["occurred_at"] = pd.to_datetime(df["occurred_at"])

    # Time-based features
    df["transaction_hour"] = df["occurred_at"].dt.hour

    df["transaction_day"] = df["occurred_at"].dt.day

    df["transaction_month"] = df["occurred_at"].dt.month

    df["transaction_day_of_week"] = (
        df["occurred_at"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["transaction_day_of_week"] >= 5
    ).astype(int)

    # Amount-based features
    df["log_amount"] = np.log1p(df["amount"])

    df["is_high_value"] = (
        df["amount"] >= 10000
    ).astype(int)

    # Time-based risk indicators
    df["is_night_transaction"] = (
        (df["transaction_hour"] >= 0) &
        (df["transaction_hour"] < 6)
    ).astype(int)

    # Account and merchant indicators
    df["is_merchant_transaction"] = (
        df["merchant_id"].notna()
    ).astype(int)

    df["is_transfer"] = (
        df["transaction_type"] == "transfer"
    ).astype(int)

    print("Feature Engineering completed!")

    return df


def main():

    df = load_transaction_data()

    df = create_features(df)

    print("\n===================================")
    print("FEATURE ENGINEERING SUMMARY")
    print("===================================")

    print(f"Total Records: {len(df)}")
    print(f"Total Features: {len(df.columns)}")

    print("\nNew Features:")

    features = [
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_day_of_week",
        "is_weekend",
        "log_amount",
        "is_high_value",
        "is_night_transaction",
        "is_merchant_transaction",
        "is_transfer"
    ]

    for feature in features:
        print(f"- {feature}")

    print("\nDataset Preview:")
    print(df.head())

    print("\nFraud Distribution:")
    print(df["is_fraud"].value_counts())


if __name__ == "__main__":
    main()