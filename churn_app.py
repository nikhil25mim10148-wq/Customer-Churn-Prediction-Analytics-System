"""
churn_app.py
============
Customer Churn Prediction & Analytics System — SINGLE-FILE VERSION.

Everything (data generation, preprocessing, model, database, predictor,
reporting, and the CLI) lives in this one file so it can be run directly
with no package/import setup:

    pip install pandas numpy scikit-learn joblib
    python churn_app.py

This is functionally identical to the modular src/ version, just flattened
into one file for convenience.
"""
import os
import sys
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "sample_customers.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.joblib")
DB_PATH = os.path.join(BASE_DIR, "churn_system.db")
LOG_PATH = os.path.join(LOG_DIR, "app.log")

TARGET_COLUMN = "churn"
CATEGORICAL_COLUMNS = ["contract_type", "internet_service", "payment_method"]
NUMERIC_COLUMNS = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
REQUIRED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]

RANDOM_STATE = 42
TEST_SIZE = 0.2

for _d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
logger = logging.getLogger("churn_app")
logger.setLevel(logging.DEBUG)
_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

_fh = logging.FileHandler(LOG_PATH)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_formatter)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(_formatter)

if not logger.handlers:
    logger.addHandler(_fh)
    logger.addHandler(_ch)


# ----------------------------------------------------------------------
# EXCEPTIONS
# ----------------------------------------------------------------------
class DataValidationError(Exception):
    """Raised when the input dataset does not match the expected schema."""


class PredictionError(Exception):
    """Raised when a prediction request is malformed or the model is unavailable."""


# ----------------------------------------------------------------------
# MODULE 1: DATA INGESTION & PREPROCESSING
# ----------------------------------------------------------------------
def generate_sample_dataset(n_rows: int = 500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Creates a synthetic but realistic telecom-style churn dataset."""
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 72, n_rows)
    monthly_charges = np.round(rng.uniform(20, 120, n_rows), 2)
    total_charges = np.round(monthly_charges * (tenure_months + 1) * rng.uniform(0.9, 1.0, n_rows), 2)
    support_calls = rng.poisson(1.5, n_rows)

    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.25, 0.20])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n_rows, p=[0.35, 0.45, 0.20])
    payment_method = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_rows)

    churn_score = (
        (contract_type == "Month-to-month") * 0.35
        + (monthly_charges > 80) * 0.2
        + (support_calls >= 3) * 0.25
        + (tenure_months < 12) * 0.2
        + rng.normal(0, 0.15, n_rows)
    )
    churn = (churn_score > 0.45).astype(int)

    df = pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_calls": support_calls,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "churn": churn,
    })
    logger.info("Generated synthetic sample dataset with %d rows", n_rows)
    return df


def load_dataset(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning("No dataset found at %s, generating a sample dataset", path)
        df = generate_sample_dataset()
        df.to_csv(path, index=False)
        return df
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed to read dataset at %s: %s", path, exc)
        raise DataValidationError(f"Could not read dataset: {exc}") from exc
    validate_schema(df)
    logger.info("Loaded dataset with %d rows from %s", len(df), path)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise DataValidationError("Dataset is empty")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().copy()
    for col in NUMERIC_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    logger.info("Cleaning removed %d duplicate rows", before - len(df))
    return df


def encode_and_scale(df: pd.DataFrame, fit: bool = True):
    df = df.copy()
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLUMNS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders, ENCODER_PATH)

        scaler = StandardScaler()
        df[NUMERIC_COLUMNS] = scaler.fit_transform(df[NUMERIC_COLUMNS])
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Fitted and saved encoders + scaler")
    else:
        encoders = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        for col in CATEGORICAL_COLUMNS:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
            )
        df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])
        logger.info("Applied saved encoders + scaler to new data")
    return df


def split_data(df: pd.DataFrame):
    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# ----------------------------------------------------------------------
# MODULE 2: MODEL TRAINING & PREDICTION
# ----------------------------------------------------------------------
class ChurnModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
        )

    def train(self, X_train, y_train):
        logger.info("Training RandomForestClassifier on %d samples", len(X_train))
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test) -> dict:
        preds = self.model.predict(X_test)
        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, preds, zero_division=0), 4),
        }
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    def feature_importance(self, feature_names) -> dict:
        importances = self.model.feature_importances_
        return dict(sorted(zip(feature_names, importances), key=lambda kv: kv[1], reverse=True))

    def predict(self, X):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        return preds, probs

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Saved trained model to %s", path)

    def load(self, path: str = MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No trained model found at {path}. Train it first.")
        self.model = joblib.load(path)
        logger.info("Loaded trained model from %s", path)
        return self


REQUIRED_PREDICT_FIELDS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def validate_prediction_input(customer: dict) -> None:
    missing = [f for f in REQUIRED_PREDICT_FIELDS if f not in customer]
    if missing:
        raise PredictionError(f"Missing required fields: {missing}")
    for col in NUMERIC_COLUMNS:
        try:
            float(customer[col])
        except (TypeError, ValueError):
            raise PredictionError(f"Field '{col}' must be numeric")


def predict_customer(customer: dict, persist: bool = True) -> dict:
    validate_prediction_input(customer)
    df = pd.DataFrame([customer])
    try:
        transformed = encode_and_scale(df, fit=False)
    except FileNotFoundError as exc:
        raise PredictionError("Model artefacts not found. Train the model first.") from exc

    model = ChurnModel()
    try:
        model.load()
    except FileNotFoundError as exc:
        raise PredictionError(str(exc)) from exc

    X = transformed[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    preds, probs = model.predict(X)

    result = {
        **{k: customer[k] for k in REQUIRED_PREDICT_FIELDS},
        "churn_prediction": int(preds[0]),
        "churn_probability": round(float(probs[0]), 4),
    }
    if persist:
        init_db()
        result["record_id"] = create_record(result)

    logger.info("Predicted churn=%s (p=%.4f) for customer", result["churn_prediction"], result["churn_probability"])
    return result


# ----------------------------------------------------------------------
# MODULE 3: STORAGE (CRUD) & REPORTING
# ----------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenure_months INTEGER NOT NULL,
    monthly_charges REAL NOT NULL,
    total_charges REAL NOT NULL,
    support_calls INTEGER NOT NULL,
    contract_type TEXT NOT NULL,
    internet_service TEXT NOT NULL,
    payment_method TEXT NOT NULL,
    churn_prediction INTEGER NOT NULL,
    churn_probability REAL NOT NULL,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def get_connection(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(SCHEMA)
    logger.info("Database initialised at %s", db_path)


def create_record(record: dict, db_path: str = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (tenure_months, monthly_charges, total_charges, support_calls,
                contract_type, internet_service, payment_method,
                churn_prediction, churn_probability, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["tenure_months"], record["monthly_charges"], record["total_charges"],
                record["support_calls"], record["contract_type"], record["internet_service"],
                record["payment_method"], record["churn_prediction"], record["churn_probability"],
                datetime.utcnow().isoformat(),
            ),
        )
        logger.info("Inserted prediction record id=%d", cur.lastrowid)
        return cur.lastrowid


def read_all_records(db_path: str = DB_PATH) -> list:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def read_record(record_id: int, db_path: str = DB_PATH):
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (record_id,)).fetchone()
        return dict(row) if row else None


def update_record(record_id: int, fields: dict, db_path: str = DB_PATH) -> bool:
    if not fields:
        return False
    columns = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [record_id]
    with get_connection(db_path) as conn:
        cur = conn.execute(f"UPDATE predictions SET {columns} WHERE id = ?", values)
        logger.info("Updated record id=%d (%d row(s) affected)", record_id, cur.rowcount)
        return cur.rowcount > 0


def delete_record(record_id: int, db_path: str = DB_PATH) -> bool:
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
        logger.info("Deleted record id=%d (%d row(s) affected)", record_id, cur.rowcount)
        return cur.rowcount > 0


def build_analytics_report() -> dict:
    records = read_all_records()
    if not records:
        return {"total_records": 0, "message": "No predictions stored yet."}
    df = pd.DataFrame(records)
    report = {
        "total_records": len(df),
        "predicted_churn_count": int(df["churn_prediction"].sum()),
        "predicted_churn_rate": round(df["churn_prediction"].mean(), 4),
        "average_churn_probability": round(df["churn_probability"].mean(), 4),
        "churn_rate_by_contract_type": df.groupby("contract_type")["churn_prediction"].mean().round(4).to_dict(),
        "average_monthly_charges_by_churn": df.groupby("churn_prediction")["monthly_charges"].mean().round(2).to_dict(),
    }
    logger.info("Generated analytics report over %d records", len(df))
    return report


def print_report(report: dict) -> None:
    if report.get("total_records", 0) == 0:
        print(report.get("message"))
        return
    print("\n--- Churn Analytics Report ---")
    print(f"Total records          : {report['total_records']}")
    print(f"Predicted churners     : {report['predicted_churn_count']}")
    print(f"Predicted churn rate   : {report['predicted_churn_rate'] * 100:.2f}%")
    print(f"Avg churn probability  : {report['average_churn_probability']:.4f}")
    print("Churn rate by contract type:")
    for k, v in report["churn_rate_by_contract_type"].items():
        print(f"  {k:<18}: {v * 100:.2f}%")
    print("Avg monthly charges (0=stay, 1=churn):")
    for k, v in report["average_monthly_charges_by_churn"].items():
        print(f"  {k}: {v}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
MENU = """
==== Customer Churn Prediction & Analytics System ====
1. Train / retrain model
2. Predict churn for a new customer
3. View analytics report
4. View all stored predictions (Read)
5. Update a stored prediction (Update)
6. Delete a stored prediction (Delete)
0. Exit
"""


def train_model():
    try:
        df = load_dataset()
        df = clean(df)
        df_transformed = encode_and_scale(df, fit=True)
        X_train, X_test, y_train, y_test = split_data(df_transformed)

        model = ChurnModel()
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        model.save()

        print("Model trained successfully.")
        print("Evaluation metrics:", metrics)
        importances = model.feature_importance(X_train.columns)
        print("Top features:", list(importances.items())[:3])
    except DataValidationError as exc:
        print(f"[ERROR] Dataset problem: {exc}")
    except Exception as exc:
        logger.exception("Unexpected error during training")
        print(f"[ERROR] Training failed: {exc}")


def predict_new_customer():
    print("Enter customer details:")
    try:
        customer = {
            "tenure_months": input("  Tenure (months): "),
            "monthly_charges": input("  Monthly charges: "),
            "total_charges": input("  Total charges: "),
            "support_calls": input("  Support calls: "),
            "contract_type": input("  Contract type [Month-to-month/One year/Two year]: "),
            "internet_service": input("  Internet service [DSL/Fiber optic/No]: "),
            "payment_method": input("  Payment method: "),
        }
        result = predict_customer(customer)
        label = "LIKELY TO CHURN" if result["churn_prediction"] == 1 else "LIKELY TO STAY"
        print(f"\nPrediction: {label} (probability={result['churn_probability']})")
        print(f"Saved as record id={result['record_id']}")
    except PredictionError as exc:
        print(f"[ERROR] {exc}")
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        print(f"[ERROR] Prediction failed: {exc}")


def view_report():
    print_report(build_analytics_report())


def view_records():
    records = read_all_records()
    if not records:
        print("No records stored yet.")
        return
    print(pd.DataFrame(records).to_string(index=False))


def update_record_cli():
    try:
        record_id = int(input("Record id to update: "))
        if not read_record(record_id):
            print("No such record.")
            return
        new_calls = input("New support_calls value (blank to skip): ").strip()
        fields = {}
        if new_calls:
            fields["support_calls"] = int(new_calls)
        ok = update_record(record_id, fields)
        print("Updated." if ok else "Nothing updated.")
    except ValueError:
        print("[ERROR] Please enter a valid integer id/value.")


def delete_record_cli():
    try:
        record_id = int(input("Record id to delete: "))
        ok = delete_record(record_id)
        print("Deleted." if ok else "No such record.")
    except ValueError:
        print("[ERROR] Please enter a valid integer id.")


def main():
    init_db()
    actions = {
        "1": train_model,
        "2": predict_new_customer,
        "3": view_report,
        "4": view_records,
        "5": update_record_cli,
        "6": delete_record_cli,
    }
    while True:
        print(MENU)
        choice = input("Choose an option: ").strip()
        if choice == "0":
            print("Goodbye.")
            sys.exit(0)
        action = actions.get(choice)
        if action:
            action()
        else:
            print("Invalid option, try again.")


if __name__ == "__main__":
    main()
