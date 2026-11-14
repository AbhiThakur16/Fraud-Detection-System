import os
import joblib

from datetime import datetime, timezone


MODEL_PATH = "artifacts/random_forest_v2.joblib"

PREPROCESSOR_PATH = "artifacts/preprocessor_v2.joblib"

OUTPUT_PATH = "artifacts/fraud_detection_model_v2.joblib"

THRESHOLD = 0.60


def main():

    print("\n===================================")
    print("SAVING BEST FRAUD DETECTION MODEL")
    print("===================================")

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    model_package = {

        "model": model,

        "preprocessor": preprocessor,

        "threshold": THRESHOLD,

        "model_name": "Random Forest",

        "model_version": "2.0",

        "created_at": datetime.now(
            timezone.utc
        ).isoformat()

    }

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    joblib.dump(
        model_package,
        OUTPUT_PATH
    )

    print("\nModel saved successfully!")

    print(f"Model: {model_package['model_name']}")

    print(f"Version: {model_package['model_version']}")

    print(f"Threshold: {THRESHOLD}")

    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":

    main()