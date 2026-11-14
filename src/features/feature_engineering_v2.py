import os
import numpy as np
import pandas as pd


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_PATH = "data/processed/fraud_transactions_v2.csv"

OUTPUT_PATH = "data/processed/fraud_features_v2.csv"


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    print("\nLoading Fraud Dataset V2...")

    df = pd.read_csv(INPUT_PATH)

    df["occurred_at"] = pd.to_datetime(
        df["occurred_at"],
        utc=True
    )

    df = df.sort_values(
        "occurred_at"
    ).reset_index(drop=True)

    print(f"Total Records: {len(df)}")

    return df


# ==========================================
# TIME-BASED FEATURES
# ==========================================

def create_time_features(df):

    print("\nCreating time-based features...")

    df["transaction_hour"] = (
        df["occurred_at"].dt.hour
    )

    df["transaction_day"] = (
        df["occurred_at"].dt.day
    )

    df["transaction_day_of_week"] = (
        df["occurred_at"].dt.dayofweek
    )

    df["transaction_month"] = (
        df["occurred_at"].dt.month
    )

    df["is_weekend"] = (
        df["transaction_day_of_week"] >= 5
    ).astype(int)

    df["is_night_transaction"] = (
        df["transaction_hour"] < 6
    ).astype(int)

    return df


# ==========================================
# AMOUNT-BASED FEATURES
# ==========================================

def create_amount_features(df):

    print("\nCreating amount-based features...")

    df["log_amount"] = np.log1p(
        df["amount"]
    )

    df["is_high_value"] = (
        df["amount"] >= 10000
    ).astype(int)

    return df


# ==========================================
# ACCOUNT BEHAVIOUR FEATURES
# ==========================================

def create_account_features(df):

    print("\nCreating account behaviour features...")

    df["account_transaction_count"] = (
        df.groupby("sender_account_id")
        .cumcount()
    )

    df["account_previous_avg_amount"] = (
        df.groupby("sender_account_id")["amount"]
        .transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    df["account_previous_max_amount"] = (
        df.groupby("sender_account_id")["amount"]
        .transform(
            lambda x: x.shift(1).expanding().max()
        )
    )

    df["amount_deviation_ratio"] = (
        df["amount"] /
        df["account_previous_avg_amount"].replace(
            0,
            np.nan
        )
    )

    df["has_previous_account_history"] = (
        df["account_transaction_count"] > 0
    ).astype(int)

    return df


# ==========================================
# TRANSACTION VELOCITY FEATURES
# ==========================================

def create_velocity_features(df):

    print("\nCreating transaction velocity features...")

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

        rolling_avg = (
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
        ] = rolling_avg.to_numpy()

    df["is_rapid_transaction"] = (
        df["account_transactions_1h"] >= 3
    ).astype(int)

    return df


# ==========================================
# MERCHANT BEHAVIOUR FEATURES
# ==========================================

def create_merchant_features(df):

    print("\nCreating merchant behaviour features...")

    df["merchant_previous_count"] = 0

    df["account_merchant_previous_count"] = 0

    merchant_mask = df["merchant_id"].notna()

    merchant_data = df.loc[merchant_mask]

    df.loc[
        merchant_mask,
        "merchant_previous_count"
    ] = (
        merchant_data
        .groupby("merchant_id")
        .cumcount()
        .to_numpy()
    )

    df.loc[
        merchant_mask,
        "account_merchant_previous_count"
    ] = (
        merchant_data
        .groupby(
            ["sender_account_id", "merchant_id"]
        )
        .cumcount()
        .to_numpy()
    )

    return df


# ==========================================
# TRANSACTION TYPE FEATURES
# ==========================================

def create_transaction_features(df):

    print("\nCreating transaction type features...")

    df["is_transfer"] = (
        df["transaction_type"] == "transfer"
    ).astype(int)

    df["is_purchase"] = (
        df["transaction_type"] == "purchase"
    ).astype(int)

    df["is_merchant_transaction"] = (
        df["merchant_id"].notna()
    ).astype(int)

    return df


# ==========================================
# SAVE DATASET
# ==========================================

def save_dataset(df):

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nDataset saved: {OUTPUT_PATH}")


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n===================================")
    print("FEATURE ENGINEERING V2")
    print("===================================")

    df = load_dataset()

    df = create_time_features(df)

    df = create_amount_features(df)

    df = create_account_features(df)

    df = create_velocity_features(df)

    df = create_merchant_features(df)

    df = create_transaction_features(df)

    print("\n===================================")
    print("FEATURE ENGINEERING SUMMARY")
    print("===================================")

    print(f"Total Records: {len(df)}")

    print(f"Total Columns: {len(df.columns)}")

    print("\nFraud Distribution:")

    print(df["is_fraud"].value_counts())

    print("\nFraud Scenario Distribution:")

    print(
        df[df["is_fraud"]]["fraud_scenario"]
        .value_counts()
    )

    print("\nFeature Preview:")

    print(
        df[
            [
                "transaction_id",
                "amount",
                "transaction_hour",
                "is_night_transaction",
                "account_transactions_1h",
                "account_transactions_24h",
                "amount_deviation_ratio",
                "is_rapid_transaction",
                "is_fraud"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nMissing Values:")

    print(
        df.isnull().sum()[
            df.isnull().sum() > 0
        ]
    )

    save_dataset(df)

    print("\nFeature Engineering V2 completed successfully!")


if __name__ == "__main__":

    main()