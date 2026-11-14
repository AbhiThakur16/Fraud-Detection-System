import os
import joblib
import numpy as np

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

DATA_PATH = "data/processed/ml_dataset.joblib"

MODEL_DIR = "artifacts"

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    print("\nLoading ML dataset...")

    dataset = joblib.load(DATA_PATH)

    X_train = dataset["X_train"]
    X_test = dataset["X_test"]

    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    print("Dataset loaded successfully!")

    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ==========================================
# MODEL EVALUATION
# ==========================================

def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

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

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\n===================================")
    print(f"{model_name} EVALUATION")
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
# TRAIN LOGISTIC REGRESSION
# ==========================================

def train_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
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
            "logistic_regression.joblib"
        )
    )

    print("Logistic Regression training complete!")

    return evaluate_model(
        model,
        X_test,
        y_test,
        "LOGISTIC REGRESSION"
    )


# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
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
            "random_forest.joblib"
        )
    )

    print("Random Forest training complete!")

    return evaluate_model(
        model,
        X_test,
        y_test,
        "RANDOM FOREST"
    )


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    X_train, X_test, y_train, y_test = load_dataset()

    results = []

    # Logistic Regression
    logistic_results = train_logistic_regression(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(logistic_results)

    # Random Forest
    forest_results = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(forest_results)

    # Comparison
    print("\n===================================")
    print("MODEL COMPARISON")
    print("===================================")

    for result in results:

        print(f"\nModel: {result['model']}")

        print(f"Precision: {result['precision']:.4f}")

        print(f"Recall: {result['recall']:.4f}")

        print(f"F1 Score: {result['f1_score']:.4f}")

        print(f"PR-AUC: {result['pr_auc']:.4f}")

    # Save comparison
    joblib.dump(
        results,
        os.path.join(
            MODEL_DIR,
            "model_comparison.joblib"
        )
    )

    print("\n===================================")

    print("MODEL TRAINING COMPLETED")

    print("===================================")

    print("\nSaved models:")

    print("1. artifacts/logistic_regression.joblib")

    print("2. artifacts/random_forest.joblib")

    print("3. artifacts/model_comparison.joblib")


if __name__ == "__main__":

    main()