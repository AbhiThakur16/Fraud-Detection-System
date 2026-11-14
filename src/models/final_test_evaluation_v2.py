
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "data/processed/ml_dataset_v2.joblib"

MODEL_PATH = "artifacts/fraud_detection_model_v2.joblib"

OUTPUT_PATH = "data/processed/final_test_results_v2.csv"


# ==========================================
# LOAD FINAL TEST DATA
# ==========================================

def load_test_data():

    print("\nLoading final test dataset...")

    dataset = joblib.load(DATA_PATH)

    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    print("\nFinal test dataset loaded!")

    print("Test samples:", len(X_test))
    print("Test features:", X_test.shape[1])

    print("Fraud transactions:", int(y_test.sum()))

    print(
        "Normal transactions:",
        int((y_test == 0).sum())
    )

    return X_test, y_test


# ==========================================
# LOAD FINAL MODEL PACKAGE
# ==========================================

def load_final_package():

    print("\nLoading final model package...")

    package = joblib.load(MODEL_PATH)

    required_keys = [
        "model",
        "preprocessor",
        "threshold"
    ]

    for key in required_keys:

        if key not in package:

            raise KeyError(
                f"Missing package component: {key}"
            )

    model = package["model"]

    preprocessor = package["preprocessor"]

    threshold = float(
        package["threshold"]
    )

    print("\nModel package loaded successfully!")

    print("Model:", package.get("model_name"))

    print("Version:", package.get("version"))

    print("Threshold:", threshold)

    print(
        "Preprocessor:",
        type(preprocessor).__name__
    )

    print(
        "Model expected features:",
        model.n_features_in_
    )

    return model, preprocessor, threshold


# ==========================================
# FINAL TEST EVALUATION
# ==========================================

def evaluate_final_model(
    model,
    preprocessor,
    X_test,
    y_test,
    threshold
):

    print("\nGenerating final test predictions...")

    # IMPORTANT:
    # X_test is already preprocessed.
    # It contains 42 transformed features.
    # Therefore, DO NOT apply preprocessor.transform()
    # again during final test evaluation.

    print(
        "\nTest dataset shape:",
        X_test.shape
    )

    if X_test.shape[1] != model.n_features_in_:

        raise ValueError(
            "Feature mismatch! "
            f"Test data has {X_test.shape[1]} features, "
            f"but model expects {model.n_features_in_}."
        )

    # Generate fraud probabilities

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Apply validation-selected threshold

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # ======================================
    # METRICS
    # ======================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

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

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # ======================================
    # CONFUSION MATRIX
    # ======================================

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    # ======================================
    # PRINT RESULTS
    # ======================================

    print("\n===================================")

    print("FINAL TEST RESULTS")

    print("===================================")

    print(f"Accuracy:  {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall:    {recall:.4f}")

    print(f"F1 Score:  {f1:.4f}")

    print(f"PR-AUC:    {pr_auc:.4f}")

    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")

    print(matrix)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix Details:")

    print("True Positives:", int(tp))

    print("False Positives:", int(fp))

    print("False Negatives:", int(fn))

    print("True Negatives:", int(tn))

    # ======================================
    # PREPARE RESULTS
    # ======================================

    results = {

        "model": package_model_name(model),

        "version": "2.0.0",

        "threshold": threshold,

        "dataset": "FINAL TEST",

        "test_samples": len(X_test),

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "pr_auc": pr_auc,

        "roc_auc": roc_auc,

        "true_positives": int(tp),

        "false_positives": int(fp),

        "false_negatives": int(fn),

        "true_negatives": int(tn)

    }

    return results


# ==========================================
# MODEL NAME
# ==========================================

def package_model_name(model):

    if model.__class__.__name__ == "RandomForestClassifier":

        return "RANDOM FOREST V2"

    return model.__class__.__name__


# ==========================================
# SAVE FINAL RESULTS
# ==========================================

def save_results(results):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    dataframe = pd.DataFrame(
        [results]
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n===================================")

    print("FINAL RESULTS SAVED")

    print("===================================")

    print(OUTPUT_PATH)


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("\n===================================")

    print("FRAUD DETECTION SYSTEM V2")

    print("PACKAGED MODEL FINAL TEST EVALUATION")

    print("===================================")

    # Load untouched test data

    X_test, y_test = load_test_data()

    # Load packaged model

    model, preprocessor, threshold = load_final_package()

    # Evaluate model

    results = evaluate_final_model(
        model,
        preprocessor,
        X_test,
        y_test,
        threshold
    )

    # Save results

    save_results(results)

    print("\n===================================")

    print("FINAL TEST EVALUATION COMPLETED")

    print("===================================")

    print("\nModel: RANDOM FOREST V2")

    print("Threshold:", threshold)

    print("\nValidation-selected threshold was used.")

    print("No final test threshold tuning performed.")

    print("\nFinal test dataset remains untouched for selection.")


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":

    main()