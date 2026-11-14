import os
import joblib
import pandas as pd

MODEL_PATH = "artifacts/fraud_detection_model_v2.joblib"


# Default values for a sample transaction
SAMPLE_TRANSACTION = {
    "amount": 1500.0,
    "transaction_type": "transfer",
    "transaction_channel": "upi",
    "channel": "upi",
    "device_type": "mobile",
    "location": "Gurugram",
    "account_age_days": 500,
    "currency": "INR",
    "transaction_status": "success",

    "transaction_hour": 14,
    "transaction_day": 10,
    "transaction_day_of_week": 2,
    "transaction_month": 11,
    "is_weekend": 0,
    "is_night_transaction": 0,

    "log_amount": 7.31,
    "is_high_value": 0,

    "account_transaction_count": 20,
    "account_previous_avg_amount": 1200.0,
    "account_previous_max_amount": 5000.0,
    "amount_deviation_ratio": 1.25,
    "has_previous_account_history": 1,

    "account_transactions_1h": 1,
    "account_transactions_24h": 3,
    "account_previous_24h_avg_amount": 1000.0,
    "is_rapid_transaction": 0,

    "merchant_previous_count": 5,
    "account_merchant_previous_count": 2,

    "is_transfer": 1,
    "is_purchase": 0,
    "is_merchant_transaction": 0
}


def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    package = joblib.load(MODEL_PATH)

    model = package["model"]
    preprocessor = package["preprocessor"]
    threshold = package["threshold"]

    return model, preprocessor, threshold


def prepare_transaction(transaction, preprocessor):

    # Get exact feature names used during training
    required_columns = list(preprocessor.feature_names_in_)

    # Start with user-provided transaction
    transaction_data = transaction.copy()

    # Default values for missing fields
    defaults = SAMPLE_TRANSACTION.copy()

    # Identify numerical and categorical columns
    numerical_columns = []
    categorical_columns = []

    for name, transformer, columns in preprocessor.transformers_:

        if name == "remainder":
            continue

        if name == "num":
            numerical_columns.extend(list(columns))

        elif name == "cat":
            categorical_columns.extend(list(columns))

    # Fill missing columns with appropriate defaults
    for column in required_columns:

        if column not in transaction_data:

            if column in defaults:
                transaction_data[column] = defaults[column]

            elif column in categorical_columns:
                transaction_data[column] = "unknown"

            else:
                transaction_data[column] = 0

    # Keep only training columns in their original order
    transaction_df = pd.DataFrame(
        [transaction_data],
        columns=required_columns
    )

    return transaction_df


def predict_transaction(transaction):

    model, preprocessor, threshold = load_model()

    # Prepare transaction
    transaction_df = prepare_transaction(
        transaction,
        preprocessor
    )

    # Apply preprocessing
    processed_data = preprocessor.transform(transaction_df)

    # Get fraud probability
    probabilities = model.predict_proba(processed_data)

    fraud_probability = float(probabilities[0][1])

    # Apply threshold
    prediction = int(fraud_probability >= threshold)

    # Prepare result
    result = {
        "status": "FRAUD ALERT" if prediction == 1 else "NORMAL",
        "prediction": prediction,
        "fraud_probability": round(fraud_probability * 100, 2),
        "threshold": threshold
    }

    return result


def main():

    print("\n===================================")
    print("REAL-TIME FRAUD DETECTION SYSTEM")
    print("===================================")

    try:

        result = predict_transaction(SAMPLE_TRANSACTION)

        print("\nTransaction Prediction")
        print("-----------------------------------")

        print(f"Status            : {result['status']}")
        print(
            f"Fraud Probability : "
            f"{result['fraud_probability']}%"
        )
        print(f"Threshold         : {result['threshold']}")

        print("-----------------------------------")

        if result["prediction"] == 1:
            print("WARNING: Potential fraudulent transaction detected!")
        else:
            print("Transaction classified as normal.")

        print("===================================")

    except Exception as error:

        print("\nPrediction failed!")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()