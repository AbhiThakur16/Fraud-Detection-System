
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "data/processed/ml_dataset_v2.joblib"

MODEL_PATH = "artifacts/random_forest_v2.joblib"

OUTPUT_PATH = "data/processed/threshold_comparison_v2.csv"

CONFIG_PATH = "artifacts/threshold_config_v2.joblib"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


# ==========================================
# LOAD VALIDATION DATA
# ==========================================

def load_validation_data():

    print("\nLoading validation dataset...")

    dataset = joblib.load(DATA_PATH)

    X_validation = dataset["X_validation"]
    y_validation = dataset["y_validation"]

    print("Validation dataset loaded!")

    print("Validation samples:", len(X_validation))
    print("Validation fraud:", int(y_validation.sum()))

    return X_validation, y_validation


# ==========================================
# LOAD RANDOM FOREST
# ==========================================

def load_model():

    print("\nLoading Random Forest V2...")

    model = joblib.load(MODEL_PATH)

    print("Random Forest loaded successfully!")

    return model


# ==========================================
# THRESHOLD EVALUATION
# ==========================================

def evaluate_thresholds(model, X_validation, y_validation):

    print("\nGenerating fraud probabilities...")

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    thresholds = np.round(
        np.arange(0.10, 0.91, 0.05),
        2
    )

    results = []

    print("\nEvaluating thresholds...")

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_validation,
            predictions
        )

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1]
        ).ravel()

        results.append({

            "threshold": float(threshold),

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1_score": f1,

            "true_positives": int(tp),

            "false_positives": int(fp),

            "false_negatives": int(fn),

            "true_negatives": int(tn)

        })

    return pd.DataFrame(results)


# ==========================================
# SELECT BEST THRESHOLD
# ==========================================

def select_best_threshold(results):

    print("\n===================================")
    print("THRESHOLD COMPARISON")
    print("===================================")

    print(
        results.to_string(index=False)
    )

    # Select threshold using maximum F1 score

    best_index = results["f1_score"].idxmax()

    best_result = results.loc[best_index]

    best_threshold = float(
        best_result["threshold"]
    )

    print("\n===================================")
    print("BEST THRESHOLD")
    print("===================================")

    print(f"Threshold: {best_threshold:.2f}")

    print(f"Precision: {best_result['precision']:.4f}")

    print(f"Recall: {best_result['recall']:.4f}")

    print(f"F1 Score: {best_result['f1_score']:.4f}")

    print("\nConfusion Matrix Details:")

    print("True Positives:", best_result["true_positives"])

    print("False Positives:", best_result["false_positives"])

    print("False Negatives:", best_result["false_negatives"])

    print("True Negatives:", best_result["true_negatives"])

    return best_threshold, best_result.to_dict()


# ==========================================
# SAVE RESULTS
# ==========================================

def save_results(results, best_threshold, best_result):

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    config = {

        "model_name": "RANDOM FOREST V2",

        "threshold": best_threshold,

        "selection_metric": "F1 Score",

        "best_validation_result": best_result,

        "dataset": "Validation",

        "final_test_used": False

    }

    joblib.dump(
        config,
        CONFIG_PATH
    )

    print("\n===================================")
    print("RESULTS SAVED")
    print("===================================")

    print("1.", OUTPUT_PATH)

    print("2.", CONFIG_PATH)


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n===================================")
    print("FRAUD DETECTION THRESHOLD TUNING V2")
    print("VALIDATION DATASET ONLY")
    print("===================================")

    X_validation, y_validation = load_validation_data()

    model = load_model()

    results = evaluate_thresholds(
        model,
        X_validation,
        y_validation
    )

    best_threshold, best_result = select_best_threshold(
        results
    )

    save_results(
        results,
        best_threshold,
        best_result
    )

    print("\n===================================")
    print("THRESHOLD TUNING COMPLETED")
    print("===================================")

    print("\nFinal test dataset remains untouched.")


if __name__ == "__main__":

    main()