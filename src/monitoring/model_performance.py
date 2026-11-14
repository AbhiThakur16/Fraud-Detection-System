from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

METRICS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_comparison_v2.csv"
)

THRESHOLD_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "threshold_comparison_v2.csv"
)


def get_model_performance():
    """
    Load saved model evaluation metrics and threshold analysis.
    """

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Model metrics file not found: {METRICS_PATH}"
        )

    df = pd.read_csv(METRICS_PATH)

    required_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "pr_auc",
        "roc_auc"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing metric columns: {missing_columns}"
        )

    metrics = []

    for _, row in df.iterrows():

        metrics.append({
            "model": str(row["model"]),
            "accuracy": round(float(row["accuracy"]) * 100, 2),
            "precision": round(float(row["precision"]) * 100, 2),
            "recall": round(float(row["recall"]) * 100, 2),
            "f1_score": round(float(row["f1_score"]) * 100, 2),
            "pr_auc": round(float(row["pr_auc"]), 4),
            "roc_auc": round(float(row["roc_auc"]), 4)
        })

    best_model = max(
        metrics,
        key=lambda item: item["f1_score"]
    )

    return {
        "total_models": len(metrics),
        "selection_metric": "F1-score",
        "best_model": best_model["model"],
        "models": metrics,
        "threshold_evaluation": get_threshold_analysis()
    }


def get_threshold_analysis():
    """
    Load threshold comparison results and identify
    the threshold with the highest F1-score.
    """

    if not THRESHOLD_PATH.exists():
        return {
            "available": False,
            "message": "Threshold comparison file not found.",
            "total_thresholds": 0,
            "best_threshold": None,
            "selection_metric": "F1-score",
            "results": []
        }

    df = pd.read_csv(THRESHOLD_PATH)

    required_columns = [
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "true_positives"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return {
            "available": False,
            "message": f"Missing columns: {missing_columns}",
            "total_thresholds": 0,
            "best_threshold": None,
            "selection_metric": "F1-score",
            "results": []
        }

    df = df.dropna(subset=required_columns).copy()

    if df.empty:
        return {
            "available": False,
            "message": "No valid threshold evaluation records found.",
            "total_thresholds": 0,
            "best_threshold": None,
            "selection_metric": "F1-score",
            "results": []
        }

    results = []

    for _, row in df.iterrows():

        results.append({
            "threshold": round(float(row["threshold"]), 2),
            "precision": round(float(row["precision"]) * 100, 2),
            "recall": round(float(row["recall"]) * 100, 2),
            "f1_score": round(float(row["f1_score"]) * 100, 2),
            "true_positives": int(row["true_positives"])
        })

    best_result = max(
        results,
        key=lambda item: item["f1_score"]
    )

    return {
        "available": True,
        "total_thresholds": len(results),
        "selection_metric": "F1-score",
        "best_threshold": best_result["threshold"],
        "best_result": best_result,
        "results": results,
        "note": (
            "Threshold selected using the saved evaluation results. "
            "Use a separate validation set for unbiased threshold "
            "selection and an untouched test set for final evaluation."
        )
    }