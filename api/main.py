
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
from datetime import datetime, date, timezone
from decimal import Decimal
import uuid
import math

from src.models.predict_fraud import predict_transaction

from src.prediction_repository import (
    save_prediction as save_prediction_to_db,
    get_prediction_history as fetch_prediction_history,
    get_prediction_summary as fetch_prediction_summary
)

from src.monitoring_repository import (
    get_monitoring_summary,
    get_hourly_trends,
    get_high_risk_predictions,
    get_prediction_confidence
)

from src.monitoring.data_drift import generate_drift_report

from src.monitoring.drift_repository import (
    get_drift_datasets
)

from src.monitoring.model_performance import (
    get_model_performance
)


# ==========================================
# CONFIGURATION
# ==========================================

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description=(
        "ML-powered fraud detection with PostgreSQL, "
        "model monitoring, confidence monitoring, "
        "data drift detection and model performance tracking."
    ),
    version="5.3.0"
)


# ==========================================
# REQUEST SCHEMA
# ==========================================

class TransactionRequest(BaseModel):
    transaction: dict[str, Any]


# ==========================================
# JSON SERIALIZATION HELPER
# ==========================================

def make_json_safe(value):

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, uuid.UUID):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    # Handle NumPy scalar values
    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except (ValueError, TypeError):
            pass

    return value


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Real-Time Fraud Detection API is running!",
        "version": "5.3.0",
        "status": "active",
        "storage": "PostgreSQL",
        "monitoring": "enabled",
        "data_drift_detection": "enabled",
        "model_performance_monitoring": "enabled",
        "documentation": "/docs"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    try:

        from src.prediction_repository import get_db_connection

        connection = get_db_connection()
        connection.close()

        database_status = "connected"

    except Exception:
        database_status = "disconnected"

    return {
        "status": "healthy",
        "database": database_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ==========================================
# FRAUD PREDICTION
# ==========================================

@app.post("/predict")
def predict(request: TransactionRequest):

    try:

        transaction = request.transaction

        if not transaction:
            raise HTTPException(
                status_code=400,
                detail="Transaction data cannot be empty."
            )

        result = predict_transaction(transaction)

        transaction_id = str(uuid.uuid4())

        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "transaction_id": transaction_id,
            "transaction_amount": transaction.get("amount", 0),
            "fraud_probability": result["fraud_probability"],
            "prediction": result["prediction"],
            "status": result["status"],
            "threshold": result["threshold"],
            "input_features": transaction
        }

        database_record_id = save_prediction_to_db(record)

        return make_json_safe({
            "success": True,
            "transaction_id": transaction_id,
            "database_record_id": database_record_id,
            "timestamp": timestamp,
            "prediction": result,
            "storage": "PostgreSQL",
            "features_stored": True
        })

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# ==========================================
# PREDICTION HISTORY
# ==========================================

@app.get("/history")
def get_prediction_history(limit: int = 100):

    try:

        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 1000."
            )

        records = fetch_prediction_history(limit)

        return make_json_safe({
            "success": True,
            "total_records": len(records),
            "history": records
        })

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"History retrieval failed: {str(error)}"
        )


# ==========================================
# HISTORY SUMMARY
# ==========================================

@app.get("/history/summary")
def history_summary():

    try:

        data = fetch_prediction_summary()

        return make_json_safe({
            "success": True,
            "summary": data
        })

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Summary retrieval failed: {str(error)}"
        )


# ==========================================
# MODEL MONITORING SUMMARY
# ==========================================

@app.get("/monitoring/summary")
def monitoring_summary():

    try:

        data = get_monitoring_summary()

        return make_json_safe({
            "success": True,
            "monitoring": data
        })

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Monitoring summary error: {str(error)}"
        )


# ==========================================
# HOURLY PREDICTION TRENDS
# ==========================================

@app.get("/monitoring/hourly-trends")
def monitoring_hourly_trends():

    try:

        data = get_hourly_trends()

        return make_json_safe({
            "success": True,
            "total_records": len(data),
            "trends": data
        })

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Hourly trends error: {str(error)}"
        )


# ==========================================
# HIGH-RISK TRANSACTION MONITORING
# ==========================================

@app.get("/monitoring/high-risk")
def monitoring_high_risk(limit: int = 20):

    try:

        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 100."
            )

        data = get_high_risk_predictions(limit)

        return make_json_safe({
            "success": True,
            "total_records": len(data),
            "high_risk_transactions": data
        })

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"High-risk monitoring error: {str(error)}"
        )


# ==========================================
# PREDICTION CONFIDENCE MONITORING
# ==========================================

@app.get("/monitoring/confidence")
def prediction_confidence():

    try:

        data = get_prediction_confidence()

        return make_json_safe({
            "success": True,
            "confidence_monitoring": data
        })

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Confidence monitoring error: {str(error)}"
        )


# ==========================================
# DATA DRIFT MONITORING
# ==========================================

@app.get("/monitoring/data-drift")
def monitoring_data_drift(limit: int = 1000):

    try:

        if limit < 1 or limit > 10000:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 10000."
            )

        reference_df, current_df = get_drift_datasets(limit)

        report = generate_drift_report(
            reference_df,
            current_df
        )

        return make_json_safe({
            "success": True,
            "reference_samples": len(reference_df),
            "current_samples": len(current_df),
            "monitoring_method": "Population Stability Index (PSI)",
            "drift_report": report
        })

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Data drift monitoring error: {str(error)}"
        )


# ==========================================
# MODEL PERFORMANCE MONITORING
# ==========================================

@app.get("/monitoring/model-performance")
def model_performance_monitoring():

    try:

        data = get_model_performance()

        return make_json_safe({
            "success": True,
            "model_performance": data
        })

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Model performance monitoring error: {str(error)}"
        )


# ==========================================
# API INFORMATION
# ==========================================

@app.get("/api-info")
def api_info():

    return {
        "project": "Real-Time Fraud Detection System",
        "version": "5.3.0",
        "model": "Random Forest V2",
        "database": "PostgreSQL",

        "endpoints": [
            "/",
            "/health",
            "/predict",
            "/history",
            "/history/summary",
            "/monitoring/summary",
            "/monitoring/hourly-trends",
            "/monitoring/high-risk",
            "/monitoring/confidence",
            "/monitoring/data-drift",
            "/monitoring/model-performance",
            "/api-info"
        ],

        "documentation": "/docs"
    }
