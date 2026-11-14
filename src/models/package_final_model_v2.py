
import os
import joblib
from datetime import datetime, timezone


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "artifacts/random_forest_v2.joblib"

PREPROCESSOR_PATH = "artifacts/preprocessor_v2.joblib"

THRESHOLD_PATH = "artifacts/threshold_config_v2.joblib"

OUTPUT_PATH = "artifacts/fraud_detection_model_v2.joblib"


# ==========================================
# BUILD FINAL MODEL PACKAGE
# ==========================================

def build_final_package():

    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Loading fitted preprocessor...")

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("Loading selected threshold...")

    threshold_config = joblib.load(THRESHOLD_PATH)

    threshold = float(
        threshold_config["threshold"]
    )

    package = {

        "model": model,

        "preprocessor": preprocessor,

        "threshold": threshold,

        "model_name": "RANDOM FOREST V2",

        "version": "2.0.0",

        "selection_metric": "PR-AUC",

        "threshold_selection_metric": "F1 Score",

        "threshold_dataset": "Validation",

        "final_test_used_for_selection": False,

        "created_at": datetime.now(
            timezone.utc
        ).isoformat()

    }

    joblib.dump(
        package,
        OUTPUT_PATH
    )

    print("\n===================================")
    print("FINAL MODEL PACKAGE CREATED")
    print("===================================")

    print("Model:", package["model_name"])

    print("Version:", package["version"])

    print("Threshold:", package["threshold"])

    print("Output:", OUTPUT_PATH)

    print("\nPackage Components:")

    print("1. Random Forest Model")

    print("2. Fitted Preprocessor")

    print("3. Selected Threshold")

    print("\nModel package created successfully!")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    build_final_package()
