import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


DATA_PATH = "data/processed/ml_dataset.joblib"

MODEL_PATH = "artifacts/logistic_regression.joblib"


def main():

    print("\n===================================")
    print("MODEL DIAGNOSTICS")
    print("===================================")

    # Load dataset
    dataset = joblib.load(DATA_PATH)

    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    print("\nTest Dataset Distribution")

    print("Normal:", np.sum(y_test == 0))
    print("Fraud:", np.sum(y_test == 1))

    # Load Logistic Regression
    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n===================================")
    print("THRESHOLD ANALYSIS")
    print("===================================")

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50
    ]

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

        print(
            f"\nThreshold: {threshold:.2f}"
            f"\nPrecision: {precision:.4f}"
            f"\nRecall: {recall:.4f}"
            f"\nF1 Score: {f1:.4f}"
        )

    print("\n===================================")
    print("PREDICTION PROBABILITY ANALYSIS")
    print("===================================")

    print(
        "Minimum:",
        probabilities.min()
    )

    print(
        "Maximum:",
        probabilities.max()
    )

    print(
        "Mean:",
        probabilities.mean()
    )

    print(
        "PR-AUC:",
        average_precision_score(
            y_test,
            probabilities
        )
    )

    # Confusion matrix at default threshold
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print("\nDefault Threshold Confusion Matrix")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\nDiagnostics completed.")


if __name__ == "__main__":

    main()
