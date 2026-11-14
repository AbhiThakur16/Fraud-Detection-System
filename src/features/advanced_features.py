import os
import pandas as pd
import numpy as np

from sqlalchemy import create_engine, URL
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_database_engine():
    """Create SQLAlchemy database engine."""

    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME")
    )

    engine = create_engine(connection_url)

    return engine


# ==========================================
# LOAD TRANSACTION DATA
# ==========================================

def load_transaction_data():

    print("\nConnecting to PostgreSQL...")

    engine = get_database_engine()

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

        df = pd.read_sql_query(query, engine)

        print("Data loaded successfully!")
        print(f"Total transactions: {len(df)}")

        return df

    finally:

        engine.dispose()


# ==========================================
# ADVANCED FEATURE ENGINEERING
# ==========================================

def create_advanced_features(df):

    print("\nCreating advanced features...")

    df = df.copy()

    # Convert timestamp to UTC
    df["occurred_at"] = pd.to_datetime(
        df["occurred_at"],
        utc=True
    )

    # Sort transactions chronologically
    df = df.sort_values(
        "occurred_at"
    ).reset_index(drop=True)

    # ======================================
    # 1. ACCOUNT TRANSACTION COUNT
    # ======================================

    df["account_transaction_count"] = (
        df.groupby("sender_account_id")
        .cumcount()
    )

    # ======================================
    # 2. ACCOUNT PREVIOUS AVERAGE AMOUNT
    # ======================================

    df["account_previous_avg_amount"] = (
        df.groupby("sender_account_id")["amount"]
        .transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    # ======================================
    # 3. ACCOUNT PREVIOUS MAXIMUM AMOUNT
    # ======================================

    df["account_previous_max_amount"] = (
        df.groupby("sender_account_id")["amount"]
        .transform(
            lambda x: x.shift(1).expanding().max()
        )
    )

    # ======================================
    # 4. AMOUNT VS ACCOUNT AVERAGE
    # ======================================

    df["amount_vs_account_avg"] = (
        df["amount"] /
        df["account_previous_avg_amount"].replace(
            0,
            np.nan
        )
    )

    # ======================================
    # 5. ACCOUNT TRANSACTION VELOCITY
    # ======================================

    df["account_transactions_1h"] = 0

    df["account_transactions_24h"] = 0

    df["account_previous_24h_avg_amount"] = np.nan

    for account_id, positions in df.groupby(
        "sender_account_id",
        sort=False
    ).indices.items():

        account_df = (
            df.iloc[positions]
            .set_index("occurred_at")
        )

        rolling_1h = (
            account_df["transaction_id"]
            .rolling(
                "1h",
                closed="left"
            )
            .count()
        )

        rolling_24h = (
            account_df["transaction_id"]
            .rolling(
                "24h",
                closed="left"
            )
            .count()
        )

        rolling_avg_24h = (
            account_df["amount"]
            .rolling(
                "24h",
                closed="left"
            )
            .mean()
        )

        df.loc[
            positions,
            "account_transactions_1h"
        ] = rolling_1h.fillna(0).to_numpy()

        df.loc[
            positions,
            "account_transactions_24h"
        ] = rolling_24h.fillna(0).to_numpy()

        df.loc[
            positions,
            "account_previous_24h_avg_amount"
        ] = rolling_avg_24h.to_numpy()

    # ======================================
    # 6. MERCHANT HISTORICAL COUNT
    # ======================================

    merchant_mask = df["merchant_id"].notna()

    df["merchant_previous_transaction_count"] = 0

    merchant_history = (
        df.loc[merchant_mask]
        .groupby("merchant_id")
        .cumcount()
    )

    df.loc[
        merchant_mask,
        "merchant_previous_transaction_count"
    ] = merchant_history.to_numpy()

    # ======================================
    # 7. PREVIOUS ACCOUNT HISTORY
    # ======================================

    df["has_previous_account_history"] = (
        df["account_transaction_count"] > 0
    ).astype(int)

    # ======================================
    # 8. HIGH TRANSACTION VELOCITY
    # ======================================

    df["high_transaction_velocity"] = (
        df["account_transactions_1h"] >= 3
    ).astype(int)

    # ======================================
    # 9. AMOUNT ABOVE PREVIOUS MAXIMUM
    # ======================================

    df["amount_above_previous_max"] = (
        (
            df["account_previous_max_amount"].notna()
        )
        &
        (
            df["amount"] >
            df["account_previous_max_amount"]
        )
    ).astype(int)

    print("Advanced features created successfully!")

    return df


# ==========================================
# DISPLAY FEATURE SUMMARY
# ==========================================

def display_feature_summary(df):

    advanced_features = [

        "account_transaction_count",

        "account_previous_avg_amount",

        "account_previous_max_amount",

        "amount_vs_account_avg",

        "account_transactions_1h",

        "account_transactions_24h",

        "account_previous_24h_avg_amount",

        "merchant_previous_transaction_count",

        "has_previous_account_history",

        "high_transaction_velocity",

        "amount_above_previous_max"
    ]

    print("\n===================================")
    print("ADVANCED FEATURE ENGINEERING")
    print("===================================")

    print(f"Total Records: {len(df)}")

    print(f"Total Columns: {len(df.columns)}")

    print("\nCreated Features:")

    for feature in advanced_features:

        print(f"- {feature}")

    print("\nFeature Preview:")

    print(
        df[
            [
                "transaction_id",
                "sender_account_id",
                "amount",
                "account_transactions_1h",
                "account_transactions_24h",
                "account_previous_avg_amount",
                "merchant_previous_transaction_count",
                "high_transaction_velocity"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nMissing Values:")

    print(
        df[advanced_features]
        .isnull()
        .sum()
    )

    print("\nFraud Distribution:")

    print(
        df["is_fraud"].value_counts()
    )


# ==========================================
# SAVE PROCESSED DATASET
# ==========================================

def save_processed_data(df):

    output_directory = "data/processed"

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    output_path = os.path.join(
        output_directory,
        "advanced_features.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n===================================")

    print("DATASET EXPORT")

    print("===================================")

    print(f"Dataset saved successfully: {output_path}")


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    try:

        # Load data
        df = load_transaction_data()

        # Create advanced features
        df = create_advanced_features(df)

        # Display summary
        display_feature_summary(df)

        # Save dataset
        save_processed_data(df)

        print("\nAdvanced Feature Engineering Completed Successfully!")

    except Exception as error:

        print(f"\nError: {error}")

        raise


if __name__ == "__main__":

    main()