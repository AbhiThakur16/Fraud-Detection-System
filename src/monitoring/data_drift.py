
import numpy as np
import pandas as pd


# ==========================================
# CONFIGURATION
# ==========================================

PSI_LOW = 0.10
PSI_MODERATE = 0.20

EPSILON = 0.0001


# ==========================================
# PSI CALCULATION
# ==========================================

def calculate_psi(reference, current, bins=10):
    """
    Calculate Population Stability Index (PSI)
    for numerical features.
    """

    reference = pd.to_numeric(
        pd.Series(reference), errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        pd.Series(current), errors="coerce"
    ).dropna()

    if len(reference) < 2 or len(current) < 2:
        return None

    # Create bins using reference distribution
    breakpoints = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, bins + 1)
        )
    )

    # Constant reference feature cannot form meaningful bins
    if len(breakpoints) < 3:
        return None

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints
    )

    reference_pct = (
        reference_counts / len(reference)
    )

    current_pct = (
        current_counts / len(current)
    )

    reference_pct = np.maximum(
        reference_pct, EPSILON
    )

    current_pct = np.maximum(
        current_pct, EPSILON
    )

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


# ==========================================
# DRIFT INTERPRETATION
# ==========================================

def interpret_drift(psi):

    if psi is None:
        return "INSUFFICIENT DATA"

    if psi < PSI_LOW:
        return "LOW DRIFT"

    elif psi < PSI_MODERATE:
        return "MODERATE DRIFT"

    return "HIGH DRIFT"


# ==========================================
# NUMERICAL FEATURE MONITORING
# ==========================================

def monitor_numerical_features(
    reference_df,
    current_df,
    bins=10
):
    """
    Compare numerical feature distributions
    between reference and current datasets.
    """

    results = []

    common_columns = set(
        reference_df.columns
    ).intersection(
        current_df.columns
    )

    for column in sorted(common_columns):

        if (
            pd.api.types.is_numeric_dtype(reference_df[column])
            and pd.api.types.is_numeric_dtype(current_df[column])
        ):

            psi = calculate_psi(
                reference_df[column],
                current_df[column],
                bins=bins
            )

            results.append({
                "feature": column,
                "feature_type": "numerical",
                "psi_score": round(psi, 4)
                if psi is not None else None,
                "drift_status": interpret_drift(psi),
                "reference_samples": int(
                    reference_df[column].count()
                ),
                "current_samples": int(
                    current_df[column].count()
                )
            })

    return results


# ==========================================
# CATEGORICAL FEATURE MONITORING
# ==========================================

def calculate_categorical_psi(reference, current):
    """
    Calculate PSI for categorical features.
    """

    reference = pd.Series(reference).dropna().astype(str)
    current = pd.Series(current).dropna().astype(str)

    if len(reference) < 2 or len(current) < 2:
        return None

    categories = sorted(
        set(reference.unique()).union(
            set(current.unique())
        )
    )

    reference_pct = (
        reference.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .values
    )

    current_pct = (
        current.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .values
    )

    reference_pct = np.maximum(
        reference_pct, EPSILON
    )

    current_pct = np.maximum(
        current_pct, EPSILON
    )

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def monitor_categorical_features(
    reference_df,
    current_df
):
    """
    Compare categorical feature distributions.
    """

    results = []

    common_columns = set(
        reference_df.columns
    ).intersection(
        current_df.columns
    )

    for column in sorted(common_columns):

        if (
            not pd.api.types.is_numeric_dtype(reference_df[column])
            and not pd.api.types.is_numeric_dtype(current_df[column])
        ):

            psi = calculate_categorical_psi(
                reference_df[column],
                current_df[column]
            )

            results.append({
                "feature": column,
                "feature_type": "categorical",
                "psi_score": round(psi, 4)
                if psi is not None else None,
                "drift_status": interpret_drift(psi),
                "reference_samples": int(
                    reference_df[column].count()
                ),
                "current_samples": int(
                    current_df[column].count()
                )
            })

    return results


# ==========================================
# COMPLETE DATA DRIFT REPORT
# ==========================================

def generate_drift_report(
    reference_df,
    current_df
):
    """
    Generate numerical and categorical
    feature drift report.
    """

    if reference_df.empty:
        raise ValueError("Reference dataset is empty.")

    if current_df.empty:
        raise ValueError("Current dataset is empty.")

    numerical_results = monitor_numerical_features(
        reference_df,
        current_df
    )

    categorical_results = monitor_categorical_features(
        reference_df,
        current_df
    )

    results = (
        numerical_results + categorical_results
    )

    report = pd.DataFrame(results)

    if report.empty:
        return {
            "total_features": 0,
            "low_drift": 0,
            "moderate_drift": 0,
            "high_drift": 0,
            "insufficient_data": 0,
            "results": []
        }

    return {
        "total_features": len(report),

        "low_drift": int(
            (report["drift_status"] == "LOW DRIFT").sum()
        ),

        "moderate_drift": int(
            (report["drift_status"] == "MODERATE DRIFT").sum()
        ),

        "high_drift": int(
            (report["drift_status"] == "HIGH DRIFT").sum()
        ),

        "insufficient_data": int(
            (report["drift_status"] == "INSUFFICIENT DATA").sum()
        ),

        "results": report.to_dict(orient="records")
    }
