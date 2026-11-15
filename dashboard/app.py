import streamlit as st
import pandas as pd
import requests
import os
import math
import joblib
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)
# ==========================================
# AUTO REFRESH
# ==========================================

st_autorefresh(
    interval=30 * 1000,
    key="fraud_dashboard_refresh"
)

# Cloud deployment: set API_URL in Streamlit Community Cloud Secrets.
# Local development falls back to the local FastAPI server.
try:
    API_URL = st.secrets.get(
        "API_URL",
        os.getenv("API_URL", "http://127.0.0.1:8000")
    ).rstrip("/")
except Exception:
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
DATA_PATH = "data/processed/fraud_features_v2.csv"

# Resolve the model package relative to the project root, not the
# terminal's current working directory. This file is dashboard/app.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "artifacts", "fraud_detection_model_v2.joblib"
)


@st.cache_resource
def get_model_threshold():
    """Read the active classification threshold from the packaged model."""
    try:
        model_package = joblib.load(MODEL_PATH)
        threshold = float(model_package.get("threshold", 0.50))
        if 0.0 <= threshold <= 1.0:
            return threshold
        st.warning("Model package contains an invalid threshold; using 0.50.")
    except Exception as error:
        st.warning(
            f"Could not read the packaged model threshold ({error}). "
            "Displaying the project default of 0.50."
        )
    return 0.50


MODEL_THRESHOLD = get_model_threshold()


# ==========================================
# PROFESSIONAL CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main Application */
.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Headings */
h1, h2, h3 {
    color: #f8fafc;
    font-weight: 700;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #1e293b,
        #172033
    );

    padding: 20px;
    border-radius: 14px;
    border: 1px solid #334155;

    min-width: 0;
    height: 100%;

    transition: all 0.2s ease;
}

/* Metric Labels */
div[data-testid="stMetricLabel"] {
    color: #cbd5e1;
    font-size: 0.95rem;

    white-space: normal !important;
    overflow-wrap: anywhere;
}

/* Metric Values */
div[data-testid="stMetricValue"] {
    color: #ffffff;

    font-size: clamp(1.35rem, 2.2vw, 2.1rem) !important;

    font-weight: 700;
    line-height: 1.3;

    white-space: normal !important;
    overflow-wrap: anywhere;
    word-break: break-word;
}

/* Metric Delta */
div[data-testid="stMetricDelta"] {
    font-size: 0.85rem;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 10px;
}

/* Input Fields */
.stTextInput input,
.stNumberInput input {
    border-radius: 8px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #20232d;
}

/* Dividers */
hr {
    border-color: #334155;
}

/* Responsive Layout */
@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1 {
        font-size: 1.8rem;
    }

    div[data-testid="stMetric"] {
        padding: 14px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# API FUNCTIONS
# ==========================================

def get_api_data(endpoint):

    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=10
        )

        if response.status_code == 200:
            return response.json()

        return None

    except requests.exceptions.RequestException:
        return None


def check_api_health():

    data = get_api_data("/health")

    return data is not None


def get_prediction_history():

    data = get_api_data("/history")

    if data and data.get("success"):

        return pd.DataFrame(data.get("history", []))

    return pd.DataFrame()



def get_monitoring_summary():

    data = get_api_data("/monitoring/summary")

    if data and data.get("success"):

        return data.get("monitoring", {})

    return None


def get_hourly_trends():

    data = get_api_data("/monitoring/hourly-trends")

    if data and data.get("success"):

        return pd.DataFrame(data.get("trends", []))

    return pd.DataFrame()


def get_high_risk_predictions():

    data = get_api_data("/monitoring/high-risk?limit=20")

    if data and data.get("success"):

        return pd.DataFrame(data.get("high_risk_transactions", []))

    return pd.DataFrame()



def get_data_drift(limit=1000):

    data = get_api_data(f"/monitoring/data-drift?limit={limit}")

    if data and data.get("success"):
        return data

    return None


def get_model_performance():

    """Fetch model evaluation metrics from the FastAPI backend."""

    data = get_api_data("/monitoring/model-performance")

    if data and data.get("success"):
        return data.get("model_performance", data)

    return None


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    return pd.DataFrame()


df = load_data()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🛡️ Fraud Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Fraud Prediction",
        "Live Prediction History",
        "Model Monitoring",
        "Data Drift Monitoring",
        "Transaction Analytics",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write("Model: Random Forest V2")
st.sidebar.write(
    f"Threshold: {MODEL_THRESHOLD:.2f} ({MODEL_THRESHOLD * 100:.0f}%)"
)
st.sidebar.write("API: FastAPI")

api_status = check_api_health()

if api_status:
    st.sidebar.success("API Status: Online")
else:
    st.sidebar.error("API Status: Offline")


# ==========================================
# DASHBOARD OVERVIEW
# ==========================================

if page == "Dashboard Overview":

    st.title("🛡️ Real-Time Fraud Detection System")

    st.write(
        "AI-powered transaction monitoring and fraud analytics dashboard."
    )

    if not df.empty and "is_fraud" in df.columns:

        total = len(df)
        fraud = int(df["is_fraud"].sum())
        normal = total - fraud

        fraud_rate = (fraud / total) * 100 if total else 0

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Transactions", f"{total:,}")
        col2.metric("Fraud Transactions", f"{fraud:,}")
        col3.metric("Normal Transactions", f"{normal:,}")
        col4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

        st.subheader("Fraud Distribution")

        distribution = pd.DataFrame({
            "Transaction Type": ["Normal", "Fraud"],
            "Count": [normal, fraud]
        })

        st.bar_chart(
            distribution.set_index("Transaction Type")
        )

        st.subheader("Recent Transactions")

        st.dataframe(
            df.tail(10),
            use_container_width=True
        )

    else:

        st.warning(
            "Dataset not found or fraud label column is missing."
        )

    st.markdown("---")

    st.subheader("Live Prediction Summary")

    history = get_prediction_history()

    if not history.empty:

        total_predictions = len(history)

        fraud_predictions = (
            int(history["prediction"].sum())
            if "prediction" in history.columns else 0
        )

        normal_predictions = total_predictions - fraud_predictions

        col1, col2, col3 = st.columns(3)

        col1.metric("Live Predictions", total_predictions)
        col2.metric("Fraud Alerts", fraud_predictions)
        col3.metric("Normal Predictions", normal_predictions)

    else:

        st.info(
            "No live prediction history available yet. "
            "Make a prediction to see live statistics."
        )


# ==========================================
# FRAUD PREDICTION
# ==========================================

elif page == "Fraud Prediction":

    st.title("🔍 Transaction Fraud Prediction")

    st.write(
        "Enter transaction information to get a prediction from the ML model."
    )

    with st.form("fraud_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            amount = st.number_input(
                "Transaction Amount (₹)",
                min_value=1.0,
                value=1500.0
            )

            transaction_type = st.selectbox(
                "Transaction Type",
                ["transfer", "purchase", "withdrawal", "deposit"]
            )

            transaction_channel = st.selectbox(
                "Transaction Channel",
                ["upi", "net_banking", "web", "atm", "mobile", "card"]
            )

            device_type = st.selectbox(
                "Device Type",
                ["mobile", "desktop", "tablet", "unknown"]
            )

            location = st.text_input(
                "Location",
                value="Gurugram"
            )

        with col2:

            account_age_days = st.number_input(
                "Account Age (Days)",
                min_value=1,
                value=500
            )

            account_previous_avg_amount = st.number_input(
                "Previous Average Transaction Amount",
                min_value=0.0,
                value=1200.0
            )

            account_previous_max_amount = st.number_input(
                "Previous Maximum Transaction Amount",
                min_value=0.0,
                value=5000.0
            )

            account_transactions_24h = st.number_input(
                "Transactions in Last 24 Hours",
                min_value=0,
                value=3
            )

            transaction_hour = st.slider(
                "Transaction Hour",
                min_value=0,
                max_value=23,
                value=14
            )

        submitted = st.form_submit_button(
            "Analyze Transaction"
        )

    if submitted:

        is_transfer = int(transaction_type == "transfer")
        is_purchase = int(transaction_type == "purchase")

        is_merchant_transaction = int(
            transaction_type in ["purchase", "withdrawal"]
        )

        previous_avg = account_previous_avg_amount

        transaction = {

            "amount": amount,
            "transaction_type": transaction_type,
            "transaction_channel": transaction_channel,
            "currency": "INR",
            "transaction_status": "success",
            "device_type": device_type,
            "location": location,
            "account_age_days": account_age_days,

            "transaction_hour": transaction_hour,
            "transaction_day": datetime.now().day,
            "transaction_day_of_week": datetime.now().weekday(),
            "transaction_month": datetime.now().month,

            "is_weekend": int(datetime.now().weekday() >= 5),

            "is_night_transaction": int(
                transaction_hour < 6 or transaction_hour >= 22
            ),

            "log_amount": math.log1p(amount),
            "is_high_value": int(amount > 10000),

            "account_transaction_count": 20,
            "account_previous_avg_amount": previous_avg,
            "account_previous_max_amount": account_previous_max_amount,

            "amount_deviation_ratio": (
                amount / previous_avg if previous_avg > 0 else 0
            ),

            "has_previous_account_history": int(previous_avg > 0),

            "account_transactions_1h": 1,
            "account_transactions_24h": account_transactions_24h,

            "account_previous_24h_avg_amount": previous_avg,

            "is_rapid_transaction": int(
                account_transactions_24h >= 5
            ),

            "merchant_previous_count": 5,
            "account_merchant_previous_count": 2,

            "is_transfer": is_transfer,
            "is_purchase": is_purchase,
            "is_merchant_transaction": is_merchant_transaction
        }

        try:

            with st.spinner("Analyzing transaction..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    json={"transaction": transaction},
                    timeout=15
                )

            if response.status_code == 200:

                result = response.json()["prediction"]

                st.markdown("---")
                st.subheader("Prediction Result")

                if result["prediction"] == 1:

                    st.error("🚨 FRAUD ALERT")

                else:

                    st.success("✅ NORMAL TRANSACTION")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Fraud Probability",
                    f"{result['fraud_probability']}%"
                )

                col2.metric(
                    "Classification Threshold",
                    f"{result['threshold'] * 100:.0f}%"
                )

                st.progress(
                    min(
                        max(result["fraud_probability"] / 100, 0.0),
                        1.0
                    )
                )

                st.info(
                    "This is a model prediction on synthetic/demo data, "
                    "not a verified banking fraud decision."
                )

                st.success(
                    "Prediction submitted successfully. "
                    "Check Live Prediction History for the record."
                )

            else:

                st.error(
                    f"API Error: {response.status_code} - {response.text}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI. "
                "Start the API server in another terminal."
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# ==========================================
# LIVE PREDICTION HISTORY
# ==========================================

elif page == "Live Prediction History":

    st.title("📡 Live Prediction History")

    st.write(
        "Monitor predictions generated by the fraud detection API."
    )

    col1, col2 = st.columns([4, 1])

    with col2:

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    history = get_prediction_history()

    if not history.empty:

        # Convert prediction values safely
        if "prediction" in history.columns:

            history["prediction"] = pd.to_numeric(
                history["prediction"],
                errors="coerce"
            ).fillna(0).astype(int)

        total = len(history)

        fraud_count = (
            int(history["prediction"].sum())
            if "prediction" in history.columns else 0
        )

        normal_count = total - fraud_count

        fraud_rate = (
            (fraud_count / total) * 100 if total > 0 else 0
        )

        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Predictions", total)
        col2.metric("Fraud Alerts", fraud_count)
        col3.metric("Normal", normal_count)
        col4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

        st.markdown("---")

        # ==================================
        # FRAUD ALERTS
        # ==================================

        st.subheader("🚨 Fraud Alerts")

        if "prediction" in history.columns:

            fraud_history = history[
                history["prediction"] == 1
            ].copy()

            if not fraud_history.empty:

                st.error(
                    f"{len(fraud_history)} potential fraud alert(s) detected."
                )

                st.dataframe(
                    fraud_history,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.success("No fraud alerts found.")

        # ==================================
        # ALL PREDICTIONS
        # ==================================

        st.markdown("---")

        st.subheader("📋 All Prediction Records")

        search_col, filter_col = st.columns(2)

        with search_col:

            search_text = st.text_input(
                "Search Transaction ID",
                placeholder="Enter transaction ID..."
            )

        with filter_col:

            status_filter = st.selectbox(
                "Filter Predictions",
                ["All", "Fraud Only", "Normal Only"]
            )

        filtered_history = history.copy()

        if status_filter == "Fraud Only":

            filtered_history = filtered_history[
                filtered_history["prediction"] == 1
            ]

        elif status_filter == "Normal Only":

            filtered_history = filtered_history[
                filtered_history["prediction"] == 0
            ]

        if search_text and "transaction_id" in filtered_history.columns:

            filtered_history = filtered_history[
                filtered_history["transaction_id"]
                .astype(str)
                .str.contains(search_text, case=False, na=False)
            ]

        st.dataframe(
            filtered_history,
            use_container_width=True,
            hide_index=True
        )

        # ==================================
        # CSV EXPORT
        # ==================================

        st.markdown("---")

        st.subheader("📥 Export Prediction History")

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download History as CSV",
            data=csv_data,
            file_name="fraud_prediction_history.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "No prediction records found. "
            "First make a transaction prediction."
        )

        st.info(
            "Your predictions will automatically appear here "
            "after successful API requests."
        )


# ==========================================
# MODEL MONITORING
# ==========================================

elif page == "Model Monitoring":

    st.title("📈 Model Monitoring")
    st.write(
        "Monitor live prediction activity, risk scores, hourly trends, "
        "and high-risk transactions from PostgreSQL."
    )

    refresh_col, note_col = st.columns([1, 4])

    with refresh_col:
        if st.button("🔄 Refresh Monitoring", use_container_width=True):
            st.rerun()

    summary = get_monitoring_summary()
    hourly = get_hourly_trends()
    high_risk = get_high_risk_predictions()

    if summary is None:
        st.error(
            "Monitoring data could not be loaded. Check that FastAPI is "
            "running and the /monitoring endpoints are available."
        )
    else:
        total_predictions = int(summary.get("total_predictions") or 0)
        fraud_alerts = int(summary.get("fraud_alerts") or 0)
        normal_transactions = int(summary.get("normal_transactions") or 0)
        average_probability = summary.get("average_fraud_probability")
        high_risk_count = int(summary.get("high_risk_transactions") or 0)
        fraud_rate = summary.get("fraud_rate")

        st.markdown("---")
        st.subheader("📊 Live Monitoring Summary")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Total Predictions", f"{total_predictions:,}")
        metric2.metric("Fraud Alerts", f"{fraud_alerts:,}")
        metric3.metric("Normal Transactions", f"{normal_transactions:,}")

        metric4, metric5, metric6 = st.columns(3)
        metric4.metric(
            "Average Fraud Probability",
            f"{float(average_probability):.2f}%"
            if average_probability is not None else "N/A"
        )
        metric5.metric("High-Risk Transactions", f"{high_risk_count:,}")
        metric6.metric(
            "Predicted Fraud Rate",
            f"{float(fraud_rate):.2f}%"
            if fraud_rate is not None else "N/A"
        )

        st.caption(
            "These are statistics from model predictions stored in the database. "
            "Predicted fraud rate is not the same as verified real-world fraud rate."
        )

        st.markdown("---")
        st.subheader("🕒 Hourly Prediction Trends — Last 24 Hours")

        if not hourly.empty:
            hourly["prediction_hour"] = pd.to_datetime(
                hourly["prediction_hour"], errors="coerce"
            )
            hourly = hourly.dropna(subset=["prediction_hour"])
            hourly = hourly.sort_values("prediction_hour")

            if not hourly.empty:
                chart_data = hourly.set_index("prediction_hour")[
                    ["total_predictions", "fraud_alerts"]
                ].astype(int)
                st.line_chart(chart_data, use_container_width=True)

                st.dataframe(
                    hourly,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No hourly prediction records are available in the last 24 hours.")
        else:
            st.info("No prediction activity found in the last 24 hours.")

        st.markdown("---")
        st.subheader("🚨 High-Risk Transactions")
        st.caption(
            "High risk here means a fraud probability of 70% or higher. "
            f"This display threshold is 70%. It is separate from the "
            f"model classification threshold of {MODEL_THRESHOLD:.2f}."
        )

        if not high_risk.empty:
            if "created_at" in high_risk.columns:
                high_risk["created_at"] = pd.to_datetime(
                    high_risk["created_at"], errors="coerce"
                )

            display_columns = [
                column for column in [
                    "transaction_id",
                    "transaction_amount",
                    "fraud_probability",
                    "status",
                    "created_at"
                ]
                if column in high_risk.columns
            ]

            st.dataframe(
                high_risk[display_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No transactions currently meet the high-risk display threshold.")

        st.markdown("---")
        st.info(
            "Monitoring currently tracks prediction volume and risk scores. "
            "It does not yet calculate data drift, concept drift, or model accuracy "
            "because verified outcome labels are not being collected for these live predictions."
        )


# ==========================================
# DATA DRIFT MONITORING
# ==========================================

elif page == "Data Drift Monitoring":

    st.title("📉 Data Drift Monitoring")
    st.write(
        "Compare the reference transaction dataset with live transaction "
        "features stored in PostgreSQL using the Population Stability Index (PSI)."
    )
    st.caption(
        "PSI interpretation used on this page: below 0.10 = Low, "
        "0.10 to below 0.25 = Moderate, and 0.25 or above = High. "
        "These are practical guidelines, not universal rules."
    )

    control_col1, control_col2 = st.columns([1, 3])

    with control_col1:
        drift_limit = st.number_input(
            "Live records to compare",
            min_value=2,
            max_value=10000,
            value=1000,
            step=100,
            key="drift_limit"
        )

    with control_col2:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Drift Report", use_container_width=True):
            st.rerun()

    # The API call is intentionally not cached. Every Streamlit rerun
    # fetches a fresh drift report from FastAPI.
    drift_data = get_data_drift(int(drift_limit))

    if drift_data is None:
        st.error(
            "Data drift report could not be loaded. Check that FastAPI is running "
            "and GET /monitoring/data-drift is available."
        )

    elif drift_data.get("drift_report") is None:
        st.warning(
            drift_data.get(
                "message",
                "No live transaction features are available yet."
            )
        )
        st.metric(
            "Reference Samples",
            f"{drift_data.get('reference_samples', 0):,}"
        )
        st.metric(
            "Current Live Samples",
            f"{drift_data.get('current_samples', 0):,}"
        )

    else:
        report = drift_data.get("drift_report", {})
        results = pd.DataFrame(report.get("results", []))

        reference_samples = int(drift_data.get("reference_samples", 0) or 0)
        current_samples = int(drift_data.get("current_samples", 0) or 0)

        if not results.empty:
            if "psi_score" not in results.columns:
                results["psi_score"] = pd.NA

            results["psi_score"] = pd.to_numeric(
                results["psi_score"], errors="coerce"
            )

            # Derive the display status from the PSI score so the dashboard
            # uses one consistent set of thresholds.
            def classify_psi(value):
                if pd.isna(value):
                    return "INSUFFICIENT DATA"
                if value < 0.10:
                    return "LOW DRIFT"
                if value < 0.25:
                    return "MODERATE DRIFT"
                return "HIGH DRIFT"

            results["dashboard_drift_status"] = results["psi_score"].apply(
                classify_psi
            )

        valid_scores = (
            results.dropna(subset=["psi_score"]).copy()
            if not results.empty else pd.DataFrame()
        )

        low_count = (
            int((valid_scores["psi_score"] < 0.10).sum())
            if not valid_scores.empty else 0
        )
        moderate_count = (
            int(
                (
                    (valid_scores["psi_score"] >= 0.10)
                    & (valid_scores["psi_score"] < 0.25)
                ).sum()
            )
            if not valid_scores.empty else 0
        )
        high_count = (
            int((valid_scores["psi_score"] >= 0.25).sum())
            if not valid_scores.empty else 0
        )
        insufficient_count = (
            int(results["psi_score"].isna().sum())
            if not results.empty else 0
        )

        st.markdown("---")
        st.subheader("📊 Drift Monitoring Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("Reference Samples", f"{reference_samples:,}")
        c2.metric("Current Live Samples", f"{current_samples:,}")
        c3.metric("Features Checked", len(results))

        c4, c5, c6, c7 = st.columns(4)
        c4.metric("🟢 Low Drift", low_count)
        c5.metric("🟡 Moderate Drift", moderate_count)
        c6.metric("🔴 High Drift", high_count)
        c7.metric("⚪ Insufficient Data", insufficient_count)

        if insufficient_count:
            st.warning(
                f"{insufficient_count} feature(s) do not have a usable PSI score. "
                "Collect more live transaction records before interpreting their drift."
            )

        st.markdown("---")
        st.subheader("🚨 Drift Alerts")

        if high_count:
            high_alerts = valid_scores[
                valid_scores["psi_score"] >= 0.25
            ].sort_values("psi_score", ascending=False)

            st.error(
                f"{high_count} feature(s) show high drift (PSI ≥ 0.25). "
                "Investigate changes in incoming transaction data."
            )
            alert_columns = [
                c for c in [
                    "feature", "feature_type", "psi_score",
                    "dashboard_drift_status", "reference_samples",
                    "current_samples"
                ] if c in high_alerts.columns
            ]
            st.dataframe(
                high_alerts[alert_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No high-drift features were identified in this report.")

        moderate_alerts = (
            valid_scores[
                (valid_scores["psi_score"] >= 0.10)
                & (valid_scores["psi_score"] < 0.25)
            ].sort_values("psi_score", ascending=False)
            if not valid_scores.empty else pd.DataFrame()
        )

        if not moderate_alerts.empty:
            st.warning(
                f"{moderate_count} feature(s) show moderate drift "
                "(PSI ≥ 0.10 and < 0.25). Keep these features under review."
            )

        st.markdown("---")
        st.subheader("📈 PSI Score by Feature")

        if not valid_scores.empty:
            chart_data = (
                valid_scores[["feature", "psi_score"]]
                .set_index("feature")
                .sort_values("psi_score", ascending=False)
            )
            st.bar_chart(chart_data, use_container_width=True)
            st.caption(
                "Higher PSI values indicate a larger difference between "
                "the reference and current feature distributions."
            )
        else:
            st.info(
                "No PSI scores can be plotted yet. One or a few live records "
                "are not enough for a meaningful distribution comparison."
            )

        st.markdown("---")
        st.subheader("🔎 Feature-wise Drift Results")

        if not results.empty:
            display_columns = [
                c for c in [
                    "feature",
                    "feature_type",
                    "psi_score",
                    "dashboard_drift_status",
                    "reference_samples",
                    "current_samples"
                ] if c in results.columns
            ]

            display_results = results[display_columns].copy()

            if "psi_score" in display_results.columns:
                display_results["psi_score"] = display_results["psi_score"].apply(
                    lambda value: round(float(value), 4)
                    if pd.notna(value) else None
                )

            st.dataframe(
                display_results,
                use_container_width=True,
                hide_index=True
            )

            csv_data = display_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Drift Report CSV",
                data=csv_data,
                file_name="fraud_data_drift_report.csv",
                mime="text/csv"
            )
        else:
            st.info("The API returned no comparable features.")

        st.markdown("---")
        st.subheader("🧭 How to Interpret PSI")

        st.dataframe(
            pd.DataFrame({
                "PSI Range": [
                    "Below 0.10",
                    "0.10 to below 0.25",
                    "0.25 or above",
                    "Not available"
                ],
                "Interpretation": [
                    "Low drift",
                    "Moderate drift",
                    "High drift — investigate",
                    "Insufficient data to calculate PSI"
                ]
            }),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "This page monitors input-feature distribution changes. It does not "
            "measure model accuracy, concept drift, or verified fraud outcomes. "
            "PSI is not reliable with only one live record; use a representative "
            "batch of live transactions."
        )


# ==========================================
# TRANSACTION ANALYTICS
# ==========================================

elif page == "Transaction Analytics":

    st.title("📊 Transaction Analytics")

    if not df.empty:

        if "transaction_type" in df.columns:

            st.subheader("Transaction Type Distribution")

            type_counts = df["transaction_type"].value_counts()

            st.bar_chart(type_counts)

        if "transaction_channel" in df.columns:

            st.subheader("Transaction Channel Analysis")

            channel_counts = df["transaction_channel"].value_counts()

            st.bar_chart(channel_counts)

        if "amount" in df.columns:

            st.subheader("Transaction Amount Distribution")

            st.line_chart(
                df["amount"].reset_index(drop=True).head(500)
            )

        if "is_fraud" in df.columns:

            st.subheader("Fraud by Transaction Channel")

            if "transaction_channel" in df.columns:

                fraud_by_channel = (
                    df.groupby("transaction_channel")["is_fraud"]
                    .sum()
                )

                st.bar_chart(fraud_by_channel)

    else:

        st.warning("Transaction dataset is not available.")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")
    st.write(
        "Compare trained fraud-detection models and analyze classification "
        "thresholds using evaluation metrics returned by the FastAPI backend."
    )

    performance = get_model_performance()

    if performance is None:
        st.error("Model performance could not be loaded from the API.")
        st.info(
            "Check that FastAPI is running and that "
            "GET /monitoring/model-performance is available."
        )
        st.code(f"{API_URL}/monitoring/model-performance")

    else:
        # Accept either the documented nested response or a direct payload.
        if isinstance(performance, dict) and "model_performance" in performance:
            performance = performance.get("model_performance") or {}

        if not isinstance(performance, dict):
            performance = {}

        models = performance.get("models", [])
        best_model = performance.get("best_model", "Not reported")
        selection_metric = performance.get("selection_metric", "Not reported")
        threshold_eval = performance.get("threshold_evaluation", {})

        # ====================================================
        # MODEL SELECTION
        # ====================================================

        st.subheader("🏆 Model Selection")
        best_col, metric_col, count_col = st.columns(3)

        # The Random Forest V2 model was selected using validation PR-AUC.
        # Keep this label explicit rather than relying on a potentially
        # inconsistent display label from an older API response.
        best_model_label = str(best_model).strip().upper()
        if "RANDOM FOREST" in best_model_label:
            best_model_label = "RANDOM FOREST V2"

        with best_col:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(145deg, #1e293b, #172033);
                    border: 1px solid #334155;
                    border-radius: 14px;
                    padding: 18px 20px;
                    min-height: 105px;
                ">
                    <div style="color:#cbd5e1;font-size:0.95rem;">Best Model</div>
                    <div style="
                        color:#ffffff;font-size:1.35rem;font-weight:750;
                        line-height:1.35;white-space:normal;overflow-wrap:anywhere;
                        margin-top:8px;
                    ">RANDOM FOREST V2</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        metric_col.metric("Selection Metric", "PR-AUC (Validation)")
        count_col.metric(
            "Models Compared",
            len(models) if isinstance(models, list) else 0
        )
        st.caption(
            "Random Forest V2 was selected using validation PR-AUC. "
            "The comparison metrics below are validation-set results."
        )

        # ====================================================
        # MODEL COMPARISON
        # ====================================================

        if isinstance(models, list) and models:
            comparison = pd.DataFrame(models)

            aliases = {
                "model": ["model", "model_name", "name"],
                "accuracy": ["accuracy", "accuracy_score"],
                "precision": ["precision", "precision_score"],
                "recall": ["recall", "recall_score"],
                "f1_score": ["f1_score", "f1", "f1-score"],
                "pr_auc": ["pr_auc", "pr-auc", "average_precision"],
                "roc_auc": ["roc_auc", "roc-auc", "roc_auc_score"],
            }

            normalized = pd.DataFrame(index=comparison.index)

            for target, candidates in aliases.items():
                source_column = next(
                    (column for column in candidates if column in comparison.columns),
                    None
                )
                normalized[target] = (
                    comparison[source_column] if source_column else pd.NA
                )

            metric_labels = {
                "accuracy": "Accuracy (%)",
                "precision": "Precision (%)",
                "recall": "Recall (%)",
                "f1_score": "F1 Score (%)",
                "pr_auc": "PR-AUC (%)",
                "roc_auc": "ROC-AUC (%)",
            }

            # Backend returns Accuracy/Precision/Recall/F1 as percentages
            # and PR-AUC/ROC-AUC as fractions. Convert AUC to percentages
            # for consistent chart/table presentation.
            for metric in metric_labels:
                normalized[metric] = pd.to_numeric(
                    normalized[metric], errors="coerce"
                )

                valid = normalized[metric].dropna()
                if not valid.empty and valid.between(0, 1).all():
                    normalized[metric] = normalized[metric] * 100

            display = normalized.rename(columns=metric_labels)
            display = display.rename(columns={"model": "Model"})

            st.divider()
            st.subheader("📊 Model Comparison — Validation Results")
            st.dataframe(
                display.round(2),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Model": st.column_config.TextColumn("Model", width="medium"),
                    "Accuracy (%)": st.column_config.NumberColumn("Accuracy (%)", format="%.2f"),
                    "Precision (%)": st.column_config.NumberColumn("Precision (%)", format="%.2f"),
                    "Recall (%)": st.column_config.NumberColumn("Recall (%)", format="%.2f"),
                    "F1 Score (%)": st.column_config.NumberColumn("F1 Score (%)", format="%.2f"),
                    "PR-AUC (%)": st.column_config.NumberColumn("PR-AUC (%)", format="%.2f"),
                    "ROC-AUC (%)": st.column_config.NumberColumn("ROC-AUC (%)", format="%.2f"),
                }
            )

            chart_columns = list(metric_labels.values())
            chart_data = display.set_index("Model")[chart_columns]
            chart_data = chart_data.dropna(how="all")

            if not chart_data.empty:
                st.subheader("📈 Performance Metrics by Model")
                # Group bars side-by-side so metrics are not visually stacked.
                st.bar_chart(
                    chart_data,
                    use_container_width=True,
                    stack=None
                )

                st.subheader("⚖️ Precision vs Recall")
                pr_data = chart_data[
                    ["Precision (%)", "Recall (%)"]
                ].dropna(how="all")

                if not pr_data.empty:
                    st.bar_chart(
                        pr_data,
                        use_container_width=True,
                        stack=None
                    )

            csv_data = display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Model Comparison CSV",
                data=csv_data,
                file_name="model_performance_report.csv",
                mime="text/csv"
            )

        else:
            st.warning(
                "The API returned no model comparison rows. "
                "Check the endpoint response before interpreting performance."
            )

        # ====================================================
        # THRESHOLD ANALYSIS
        # ====================================================

        st.divider()
        st.header("🎯 Threshold Analysis & Optimization")

        if isinstance(threshold_eval, dict) and threshold_eval.get("available"):
            threshold_results = pd.DataFrame(
                threshold_eval.get("results", [])
            )

            best_threshold = threshold_eval.get("best_threshold")
            best_result = threshold_eval.get("best_result") or {}

            required_columns = [
                "threshold", "precision", "recall", "f1_score"
            ]

            if threshold_results.empty:
                st.warning(
                    "Threshold evaluation is available, but no threshold rows were returned."
                )
            else:
                missing_columns = [
                    column for column in required_columns
                    if column not in threshold_results.columns
                ]

                if missing_columns:
                    st.warning(
                        f"Threshold data is missing required columns: {missing_columns}"
                    )
                else:
                    for column in required_columns:
                        threshold_results[column] = pd.to_numeric(
                            threshold_results[column],
                            errors="coerce"
                        )

                    threshold_results = (
                        threshold_results
                        .dropna(subset=required_columns)
                        .sort_values("threshold")
                    )

                    if threshold_results.empty:
                        st.warning("No valid numeric threshold results were returned.")
                    else:
                        # ------------------------------------------------
                        # BEST THRESHOLD KPI CARDS
                        # ------------------------------------------------

                        st.subheader("🏆 Best Threshold Performance")

                        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

                        kpi1.metric(
                            "Best Threshold",
                            f"{float(best_threshold):.2f}"
                            if best_threshold is not None else "N/A"
                        )
                        kpi2.metric(
                            "Precision",
                            f"{float(best_result.get('precision', 0)):.2f}%"
                        )
                        kpi3.metric(
                            "Recall",
                            f"{float(best_result.get('recall', 0)):.2f}%"
                        )
                        kpi4.metric(
                            "Best F1-score",
                            f"{float(best_result.get('f1_score', 0)):.2f}%"
                        )

                        st.caption(
                            "The displayed best threshold is the threshold with "
                            "the highest F1-score in the saved evaluation results."
                        )

                        # ------------------------------------------------
                        # PRECISION AND RECALL CHART
                        # ------------------------------------------------

                        st.divider()
                        st.subheader("📈 Precision vs Recall by Threshold")

                        pr_chart = threshold_results[
                            ["threshold", "precision", "recall"]
                        ].set_index("threshold")

                        pr_chart = pr_chart.rename(
                            columns={
                                "precision": "Precision (%)",
                                "recall": "Recall (%)"
                            }
                        )

                        st.line_chart(
                            pr_chart,
                            use_container_width=True
                        )

                        st.caption(
                            "As the classification threshold increases, precision "
                            "may improve while recall may decrease."
                        )

                        # ------------------------------------------------
                        # F1 SCORE CHART
                        # ------------------------------------------------

                        st.divider()
                        st.subheader("🎯 F1-score vs Classification Threshold")

                        f1_chart = threshold_results[
                            ["threshold", "f1_score"]
                        ].set_index("threshold")

                        f1_chart = f1_chart.rename(
                            columns={"f1_score": "F1-score (%)"}
                        )

                        st.line_chart(
                            f1_chart,
                            use_container_width=True
                        )

                        if best_threshold is not None:
                            st.success(
                                f"Highest saved F1-score: "
                                f"{float(best_result.get('f1_score', 0)):.2f}% "
                                f"at threshold {float(best_threshold):.2f}."
                            )

                        # ------------------------------------------------
                        # FULL THRESHOLD TABLE
                        # ------------------------------------------------

                        st.divider()
                        st.subheader("📋 Complete Threshold Comparison")

                        threshold_display = threshold_results.rename(
                            columns={
                                "threshold": "Threshold",
                                "precision": "Precision (%)",
                                "recall": "Recall (%)",
                                "f1_score": "F1-score (%)",
                                "true_positives": "True Positives"
                            }
                        ).copy()

                        numeric_columns = [
                            column for column in [
                                "Threshold",
                                "Precision (%)",
                                "Recall (%)",
                                "F1-score (%)"
                            ]
                            if column in threshold_display.columns
                        ]

                        threshold_display[numeric_columns] = (
                            threshold_display[numeric_columns].round(2)
                        )

                        if best_threshold is not None:
                            threshold_display["Best Threshold"] = (
                                threshold_display["Threshold"]
                                == float(best_threshold)
                            )

                        st.dataframe(
                            threshold_display,
                            use_container_width=True,
                            hide_index=True
                        )

                        threshold_csv = threshold_display.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="📥 Download Threshold Analysis CSV",
                            data=threshold_csv,
                            file_name="threshold_analysis_report.csv",
                            mime="text/csv"
                        )

                        # ------------------------------------------------
                        # RECOMMENDATION AND LIMITATION
                        # ------------------------------------------------

                        st.divider()
                        st.subheader("💡 Threshold Recommendation")

                        if best_threshold is not None:
                            st.info(
                                f"**Best threshold in the saved evaluation:** "
                                f"{float(best_threshold):.2f}\n\n"
                                f"**Precision:** "
                                f"{float(best_result.get('precision', 0)):.2f}%\n\n"
                                f"**Recall:** "
                                f"{float(best_result.get('recall', 0)):.2f}%\n\n"
                                f"**F1-score:** "
                                f"{float(best_result.get('f1_score', 0)):.2f}%"
                            )

                        st.success(
                            "Threshold selection was performed on the validation set. "
                            "The final test set remained separate and was used only "
                            "for the final evaluation."
                        )

        else:
            st.info(
                "Threshold analysis is not currently available from the FastAPI backend."
            )

        # ====================================================
        # EVALUATION NOTES
        # ====================================================

        st.divider()
        st.subheader("ℹ️ Evaluation Notes")

        st.info(
            "Metrics on this page are sourced from the FastAPI model-performance "
            "endpoint. The project uses synthetic transaction data; these results "
            "should not be interpreted as real-world production performance."
        )

        st.warning(
            "Use a validation set for model and threshold selection, then report "
            "final metrics on an untouched test set. Do not mix metrics measured "
            "at different classification thresholds."
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Real-Time Fraud Detection & Transaction Analytics System | "
    "Machine Learning Project"
)