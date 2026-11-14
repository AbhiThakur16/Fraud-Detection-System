import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "data/processed/ml_dataset_v2.joblib"

MODEL_PATH = "artifacts/random_forest_v2.joblib"


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("\n===================================")
    print("RANDOM FOREST MODEL DIAGNOSTICS V2")
    print("===================================")

    # Load dataset
    dataset = joblib.load(DATA_PATH)

    X_test = dataset["X_test"]

    y_test = dataset["y_test"]

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Get fraud probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nTest Dataset:")

    print(f"Total Transactions: {len(y_test)}")

    print(f"Fraud Transactions: {np.sum(y_test == 1)}")

    print("\nPR-AUC:")

    print(
        average_precision_score(
            y_test,
            probabilities
        )
    )

    # Threshold analysis
    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]

    results = []

    print("\n===================================")
    print("THRESHOLD COMPARISON")
    print("===================================")

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).ravel()

        results.append({

            "threshold": threshold,

            "precision": precision,

            "recall": recall,

            "f1_score": f1,

            "true_positives": tp,

            "false_positives": fp,

            "false_negatives": fn,

            "true_negatives": tn

        })

        print(f"\nThreshold: {threshold:.2f}")

        print(f"Precision: {precision:.4f}")

        print(f"Recall: {recall:.4f}")

        print(f"F1 Score: {f1:.4f}")

        print(f"True Positives: {tp}")

        print(f"False Positives: {fp}")

        print(f"False Negatives: {fn}")

    # Save results
    comparison = pd.DataFrame(results)

    comparison.to_csv(
        "data/processed/threshold_comparison_v2.csv",
        index=False
    )

    print("\n===================================")

    print("THRESHOLD ANALYSIS COMPLETED")

    print("===================================")

    print(
        "\nSaved: data/processed/threshold_comparison_v2.csv"
    )


if __name__ == "__main__":

    main()