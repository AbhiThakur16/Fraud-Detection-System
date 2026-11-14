
import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

MODEL_DIR = "artifacts"
PROCESSED_DIR = "data/processed"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    print("\nLoading ML Dataset V2...")

    dataset = joblib.load(DATA_PATH)

    X_train = dataset["X_train"]
    X_validation = dataset["X_validation"]

    y_train = dataset["y_train"]
    y_validation = dataset["y_validation"]

    print("\nDataset loaded successfully!")

    print("\nDataset Split:")
    print("-----------------------------------")

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")

    print("\nTraining fraud:", int(y_train.sum()))
    print("Training normal:", int((y_train == 0).sum()))

    print("\nValidation fraud:", int(y_validation.sum()))
    print("Validation normal:", int((y_validation == 0).sum()))

    print("\nFinal test dataset is intentionally untouched.")

    return (
        X_train,
        X_validation,
        y_train,
        y_validation
    )


# ==========================================
# MODEL EVALUATION
# ==========================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
    model_name
):

    predictions = model.predict(X_validation)

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

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

    pr_auc = average_precision_score(
        y_validation,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities
    )

    matrix = confusion_matrix(
        y_validation,
        predictions
    )

    print("\n===================================")
    print(f"{model_name} VALIDATION RESULTS")
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
            y_validation,
            predictions,
            zero_division=0
        )
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    }


# ==========================================
# LOGISTIC REGRESSION
# ==========================================

def train_logistic_regression(
    X_train,
    X_validation,
    y_train,
    y_validation
):

    print("\nTraining Logistic Regression V2...")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=1.0,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        model,
        os.path.join(
            MODEL_DIR,
            "logistic_regression_v2.joblib"
        )
    )

    print("\nLogistic Regression V2 training completed!")

    return evaluate_model(
        model,
        X_validation,
        y_validation,
        "LOGISTIC REGRESSION V2"
    )


# ==========================================
# RANDOM FOREST
# ==========================================

def train_random_forest(
    X_train,
    X_validation,
    y_train,
    y_validation
):

    print("\nTraining Random Forest V2...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        model,
        os.path.join(
            MODEL_DIR,
            "random_forest_v2.joblib"
        )
    )

    print("\nRandom Forest V2 training completed!")

    return evaluate_model(
        model,
        X_validation,
        y_validation,
        "RANDOM FOREST V2"
    )


# ==========================================
# MODEL COMPARISON
# ==========================================

def compare_models(results):

    print("\n===================================")
    print("MODEL COMPARISON V2")
    print("VALIDATION DATASET")
    print("===================================")

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="pr_auc",
        ascending=False
    ).reset_index(drop=True)

    print(
        comparison[
            [
                "model",
                "precision",
                "recall",
                "f1_score",
                "pr_auc",
                "roc_auc"
            ]
        ].to_string(index=False)
    )

    # Save validation comparison

    comparison.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "model_comparison_v2.csv"
        ),
        index=False
    )

    joblib.dump(
        results,
        os.path.join(
            MODEL_DIR,
            "model_comparison_v2.joblib"
        )
    )

    # Select best model using validation PR-AUC

    best_model = comparison.iloc[0]

    print("\n===================================")
    print("BEST MODEL SELECTION")
    print("===================================")

    print("Selection Metric: PR-AUC")
    print("Best Model:", best_model["model"])
    print(f"Best PR-AUC: {best_model['pr_auc']:.4f}")

    print("\nValidation comparison saved successfully!")

    return best_model["model"]


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("\n===================================")
    print("FRAUD DETECTION MODEL TRAINING V2")
    print("70% TRAIN / 15% VALIDATION / 15% TEST")
    print("===================================")

    (
        X_train,
        X_validation,
        y_train,
        y_validation
    ) = load_dataset()

    results = []

    # Logistic Regression

    logistic_results = train_logistic_regression(
        X_train,
        X_validation,
        y_train,
        y_validation
    )

    results.append(logistic_results)

    # Random Forest

    forest_results = train_random_forest(
        X_train,
        X_validation,
        y_train,
        y_validation
    )

    results.append(forest_results)

    # Compare models

    best_model = compare_models(results)

    print("\n===================================")
    print("MODEL TRAINING V2 COMPLETED")
    print("===================================")

    print("\nSaved Models:")

    print("1. artifacts/logistic_regression_v2.joblib")
    print("2. artifacts/random_forest_v2.joblib")
    print("3. artifacts/model_comparison_v2.joblib")

    print("\nBest Model:", best_model)

    print("\nFinal test dataset was NOT used.")

    print("\nNext Step: Validation-based threshold tuning.")


if __name__ == "__main__":

    main()
