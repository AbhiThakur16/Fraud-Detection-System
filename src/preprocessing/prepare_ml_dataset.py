import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_PATH = "data/processed/advanced_features.csv"

OUTPUT_DIR = "data/processed"

MODEL_DIR = "artifacts"


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    print("\nLoading processed dataset...")

    df = pd.read_csv(INPUT_PATH)

    df["occurred_at"] = pd.to_datetime(
        df["occurred_at"],
        utc=True
    )

    print(f"Total records: {len(df)}")

    return df


# ==========================================
# FEATURE CREATION
# ==========================================

def create_ml_features(df):

    print("\nCreating ML features...")

    df = df.copy()

    # Time-based features
    df["transaction_hour"] = df["occurred_at"].dt.hour

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

    # Amount-based features
    df["log_amount"] = __import__("numpy").log1p(
        df["amount"]
    )

    df["is_high_value"] = (
        df["amount"] >= 10000
    ).astype(int)

    return df


# ==========================================
# SELECT FEATURES AND TARGET
# ==========================================

def select_features(df):

    print("\nSelecting model features...")

    numerical_features = [

        "amount",
        "log_amount",
        "transaction_hour",
        "transaction_day_of_week",
        "transaction_month",
        "is_weekend",
        "is_night_transaction",
        "is_high_value",
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

    categorical_features = [

        "transaction_type",
        "transaction_channel",
        "currency",
        "transaction_status"
    ]

    selected_features = (
        numerical_features +
        categorical_features
    )

    X = df[selected_features].copy()

    y = df["is_fraud"].astype(int)

    return X, y, numerical_features, categorical_features


# ==========================================
# CHRONOLOGICAL TRAIN TEST SPLIT
# ==========================================

def split_dataset(df, X, y):

    print("\nSplitting dataset chronologically...")

    sorted_indices = (
        df["occurred_at"]
        .sort_values()
        .index
    )

    split_position = int(len(sorted_indices) * 0.80)

    train_indices = sorted_indices[:split_position]

    test_indices = sorted_indices[split_position:]

    X_train = X.loc[train_indices]

    X_test = X.loc[test_indices]

    y_train = y.loc[train_indices]

    y_test = y.loc[test_indices]

    print(f"Training records: {len(X_train)}")

    print(f"Testing records: {len(X_test)}")

    print(f"Training fraud cases: {y_train.sum()}")

    print(f"Testing fraud cases: {y_test.sum()}")

    return X_train, X_test, y_train, y_test


# ==========================================
# PREPROCESSING PIPELINE
# ==========================================

def build_preprocessor(
    numerical_features,
    categorical_features
):

    numerical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),

            (
                "scaler",
                StandardScaler()
            )

        ]
    )

    categorical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )

        ]
    )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )

        ]
    )

    return preprocessor


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load dataset
    df = load_dataset()

    # Create features
    df = create_ml_features(df)

    # Select features
    X, y, numerical, categorical = select_features(df)

    # Chronological split
    X_train, X_test, y_train, y_test = split_dataset(
        df,
        X,
        y
    )

    # Create preprocessing pipeline
    preprocessor = build_preprocessor(
        numerical,
        categorical
    )

    print("\nFitting preprocessing pipeline...")

    # Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Transform test data without fitting
    X_test_processed = preprocessor.transform(
        X_test
    )

    # Save preprocessor
    joblib.dump(
        preprocessor,
        os.path.join(MODEL_DIR, "preprocessor.joblib")
    )

    # Save processed datasets
    joblib.dump(
        {
            "X_train": X_train_processed,
            "X_test": X_test_processed,
            "y_train": y_train.to_numpy(),
            "y_test": y_test.to_numpy()
        },
        os.path.join(OUTPUT_DIR, "ml_dataset.joblib")
    )

    print("\n===================================")
    print("ML DATASET PREPARATION COMPLETE")
    print("===================================")

    print(f"Training shape: {X_train_processed.shape}")

    print(f"Testing shape: {X_test_processed.shape}")

    print(f"Training fraud cases: {y_train.sum()}")

    print(f"Testing fraud cases: {y_test.sum()}")

    print("\nSaved files:")

    print("1. data/processed/ml_dataset.joblib")

    print("2. artifacts/preprocessor.joblib")


if __name__ == "__main__":

    main()