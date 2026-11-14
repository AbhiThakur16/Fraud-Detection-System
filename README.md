# Real-Time Fraud Detection & Transaction Analytics System

An end-to-end Machine Learning application designed to detect potentially fraudulent financial transactions and provide real-time transaction analytics, fraud monitoring, and model performance insights.

## 🚀 Project Overview

This project combines Machine Learning, FastAPI, PostgreSQL, and Streamlit to build a fraud detection and transaction analytics platform.

The system uses a trained Random Forest classifier to predict whether a transaction is potentially fraudulent. It also provides transaction history, fraud alerts, model monitoring, and data drift analysis.

## ✨ Key Features

- Machine Learning-based fraud detection
- Real-time fraud prediction through FastAPI
- Interactive Streamlit dashboard
- PostgreSQL database integration
- Transaction prediction history
- Fraud alert monitoring
- Transaction analytics and visualizations
- Data drift monitoring using PSI
- Model performance monitoring
- CSV export functionality

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Backend API | FastAPI |
| Frontend Dashboard | Streamlit |
| Database | PostgreSQL |
| Model Serialization | Joblib |
| API Server | Uvicorn |
| Database Integration | SQLAlchemy, Psycopg2 |

## 🏗️ System Architecture

```text
                 Transaction Input
                        |
                        v
                Streamlit Dashboard
                        |
                        v
                  FastAPI Backend
                        |
                        v
               Feature Preprocessing
                        |
                        v
               Random Forest Model
                        |
                        v
                Fraud Prediction
                        |
              +---------+---------+
              |                   |
              v                   v
        PostgreSQL           API Response
              |
              v
     Transaction History
              |
              v
      Monitoring & Analytics
```

## 🤖 Machine Learning Model

The project uses a Random Forest classifier trained on a synthetic transaction dataset.

### Dataset Information

- Total transactions: 50,000
- Normal transactions: 49,000
- Fraudulent transactions: 1,000
- Fraud ratio: 2%
- Dataset type: Synthetic

### Model Evaluation

The final model was evaluated on an untouched test dataset.

| Metric | Test Result |
|---|---:|
| Accuracy | 99.48% |
| Precision | 80.71% |
| Recall | 90.40% |
| F1 Score | 85.28% |
| PR-AUC | 0.9178 |
| ROC-AUC | 0.9968 |

Classification threshold: **0.50**

**Note:** These results are from a synthetic dataset and should not be interpreted as guaranteed performance on real-world financial transactions.

## 📊 Dashboard Pages

The Streamlit application contains seven main pages:

1. Dashboard Overview
2. Fraud Prediction
3. Live Prediction History
4. Model Monitoring
5. Data Drift Monitoring
6. Transaction Analytics
7. Model Performance

## 📁 Project Structure

```text
Fraud-Detection-System/
|
|-- api/
|   |-- main.py
|
|-- artifacts/
|   |-- fraud_detection_model_v2.joblib
|
|-- dashboard/
|   |-- app.py
|
|-- database/
|   |-- sql/
|
|-- docs/
|
|-- src/
|   |-- features/
|   |-- ingestion/
|   |-- models/
|   |-- monitoring/
|   |-- preprocessing/
|   |-- db_connection.py
|   |-- prediction_repository.py
|   |-- monitoring_repository.py
|
|-- tests/
|
|-- .env.example
|-- .gitignore
|-- requirements.txt
|-- README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AbhiThakur16/Fraud-Detection-System.git
cd Fraud-Detection-System
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create your local `.env` file:

```powershell
Copy-Item .env.example .env
```

Update the database configuration in `.env`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fraud_detection_db
DB_USER=postgres
DB_PASSWORD=your_password
```

Use your own PostgreSQL credentials. Never upload your actual `.env` file.

### 5. Configure PostgreSQL

Create a PostgreSQL database named:

```sql
CREATE DATABASE fraud_detection_db;
```

Execute the database schema scripts from `database/sql/` in the appropriate order.

### 6. Start the FastAPI Backend

Open the first terminal:

```bash
uvicorn api.main:app --reload
```

API URL:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### 7. Start the Streamlit Dashboard

Open a second terminal:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

## 🔗 API Endpoints

| Endpoint | Description |
|---|---|
| `/` | API home |
| `/health` | Health check |
| `/api-info` | API information |
| `/docs` | Swagger documentation |
| `/predict` | Fraud prediction |
| `/history` | Prediction history |
| `/history/summary` | History summary |
| `/monitoring/summary` | Monitoring summary |
| `/monitoring/hourly-trends` | Hourly trends |
| `/monitoring/high-risk` | High-risk transactions |
| `/monitoring/confidence` | Prediction confidence |
| `/monitoring/data-drift` | Data drift analysis |
| `/monitoring/model-performance` | Model performance |

## 📈 Model Monitoring

The application includes monitoring capabilities for:

- Prediction trends
- Fraud and normal prediction counts
- High-risk transactions
- Prediction confidence
- Feature distribution changes
- Population Stability Index (PSI)
- Model performance metrics

Data drift indicators are preliminary monitoring signals. They do not independently establish model degradation.

## 🔐 Security

- Database credentials are stored in environment variables.
- `.env` is excluded through `.gitignore`.
- Virtual environments and generated datasets are excluded from version control.
- Database backups are not included in the public repository.

## ⚠️ Limitations

- The training dataset is synthetic.
- Model performance on real financial transactions may differ.
- Real-world deployment requires additional validation, security controls, and monitoring.
- Live prediction outputs are model predictions, not independently verified fraud labels.

## 🔮 Future Improvements

- Integration with real-time transaction streams
- Advanced anomaly detection
- Model retraining pipeline
- Authentication and role-based access control
- Enhanced fraud investigation tools
- Production deployment and cloud monitoring

## 👨‍💻 Author

**Abhi Thakur**

B.Tech Computer Science & Engineering (Data Science)

GitHub: [AbhiThakur16](https://github.com/AbhiThakur16)

---

If you find this project interesting, consider giving the repository a star! ⭐