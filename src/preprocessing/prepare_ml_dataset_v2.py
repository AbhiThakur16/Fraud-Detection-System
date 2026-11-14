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

INPUT_PATH = "data/processed/fraud_features_v2.csv"

OUTPUT_DIR = "data/processed"
MODEL_DIR = "artifacts"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    print("\nLoading Fraud Dataset V2...")

    df = pd.read_csv(INPUT_PATH)

    df["occurred_at"] = pd.to_datetime(
        df["occurred_at"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=["occurred_at", "is_fraud"]
    )

    df = df.sort_values(
        "occurred_at"
    ).reset_index(drop=True)

    df["is_fraud"] = df["is_fraud"].astype(int)

    print(f"Total Records: {len(df):,}")

    print("\nOverall Fraud Distribution:")
    print(df["is_fraud"].value_counts())

    return df


# ==========================================
# FEATURE SELECTION
# ==========================================

def select_features(df):

    print("\nSelecting ML Features...")

    numerical_features = [

        "amount",
        "log_amount",
        "transaction_hour",
        "transaction_day",
        "transaction_day_of_week",
        "transaction_month",
        "is_weekend",
        "is_night_transaction",
        "is_high_value",
        "account_transaction_count",
        "account_previous_avg_amount",
        "account_previous_max_amount",
        "amount_deviation_ratio",
        "has_previous_account_history",
        "account_transactions_1h",
        "account_transactions_24h",
        "account_previous_24h_avg_amount",
        "is_rapid_transaction",
        "merchant_previous_count",
        "account_merchant_previous_count",
        "is_transfer",
        "is_purchase",
        "is_merchant_transaction"

    ]

    categorical_features = [

        "transaction_type",
        "transaction_channel",
        "currency",
        "transaction_status"

    ]

    selected_features = (
        numerical_features + categorical_features
    )

    missing = [
        col for col in selected_features + ["is_fraud"]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required dataset columns: {missing}"
        )

    X = df[selected_features].copy()

    y = df["is_fraud"].copy()

    print(f"Numerical Features: {len(numerical_features)}")
    print(f"Categorical Features: {len(categorical_features)}")
    print(f"Total Features: {len(selected_features)}")

    return X, y, numerical_features, categorical_features


# ==========================================
# CHRONOLOGICAL THREE-WAY SPLIT
# ==========================================

def split_dataset(X, y):

    print("\nCreating chronological dataset splits...")

    total = len(X)

    train_position = int(total * TRAIN_RATIO)

    validation_position = int(
        total * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    X_train = X.iloc[:train_position].copy()
    y_train = y.iloc[:train_position].copy()

    X_validation = X.iloc[
        train_position:validation_position
    ].copy()

    y_validation = y.iloc[
        train_position:validation_position
    ].copy()

    X_test = X.iloc[validation_position:].copy()
    y_test = y.iloc[validation_position:].copy()

    splits = {
        "Training": y_train,
        "Validation": y_validation,
        "Final Testing": y_test
    }

    print("\nDataset Split Summary")

    for name, labels in splits.items():

        fraud = int(labels.sum())
        normal = len(labels) - fraud

        fraud_rate = (
            fraud / len(labels) * 100
            if len(labels) else 0
        )

        print(f"\n{name}")
        print(f"Total: {len(labels):,}")
        print(f"Normal: {normal:,}")
        print(f"Fraud: {fraud:,}")
        print(f"Fraud Rate: {fraud_rate:.2f}%")

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


# ==========================================
# BUILD PREPROCESSING PIPELINE
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
                "num",
                numerical_pipeline,
                numerical_features
            ),

            (
                "cat",
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

    # Select features
    X, y, numerical, categorical = select_features(df)

    # Three-way chronological split
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = split_dataset(X, y)

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(
        numerical,
        categorical
    )

    print("\nFitting preprocessing pipeline on training data...")

    # Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Validation and test only transformed
    X_validation_processed = preprocessor.transform(
        X_validation
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print("\nPreprocessing completed successfully!")

    print("Training Shape:", X_train_processed.shape)
    print("Validation Shape:", X_validation_processed.shape)
    print("Testing Shape:", X_test_processed.shape)

    # Save preprocessor
    joblib.dump(
        preprocessor,
        os.path.join(
            MODEL_DIR,
            "preprocessor_v2.joblib"
        )
    )

    # Save complete dataset
    dataset = {

        "X_train": X_train_processed,
        "X_validation": X_validation_processed,
        "X_test": X_test_processed,

        "y_train": y_train.to_numpy(),
        "y_validation": y_validation.to_numpy(),
        "y_test": y_test.to_numpy(),

        "feature_names": preprocessor.get_feature_names_out(),

        "numerical_features": numerical,
        "categorical_features": categorical,

        "split_strategy": "chronological",
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO

    }

    output_path = os.path.join(
        OUTPUT_DIR,
        "ml_dataset_v2.joblib"
    )

    joblib.dump(
        dataset,
        output_path
    )

    print("\n======================================")
    print("ML DATASET V2 PREPARATION COMPLETED")
    print("======================================")

    print(f"Training: {len(y_train):,}")
    print(f"Validation: {len(y_validation):,}")
    print(f"Final Testing: {len(y_test):,}")

    print("\nSaved Files:")
    print(output_path)
    print("artifacts/preprocessor_v2.joblib")


if __name__ == "__main__":
    main()