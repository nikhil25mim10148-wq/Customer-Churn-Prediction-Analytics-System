# Customer Churn Prediction & Analytics System

A machine learning system that predicts whether a telecom customer is likely
to churn (cancel their subscription), stores every prediction, and produces
analytics reports over the stored history. Built as a VITyarthi course
project for Machine Learning / AI.

## Overview

Customer churn — customers leaving for a competitor — is one of the most
expensive problems for subscription-based businesses. This project builds a
small but complete pipeline that:

1. Ingests and cleans customer usage/billing data
2. Trains a classification model (Random Forest) to predict churn
3. Serves predictions for new customers through a CLI
4. Persists every prediction to a local database with full CRUD support
5. Generates churn analytics (overall rate, rate by contract type, etc.)

## Features

- **Synthetic data generator** — creates a realistic sample dataset on first
  run, so the project works out of the box with no external dataset needed
- **Data cleaning & preprocessing** — duplicate removal, missing-value
  handling, categorical encoding, feature scaling
- **Churn classification model** — Random Forest classifier with
  train/evaluate/save/load lifecycle and feature-importance reporting
- **Prediction pipeline** — validates input, reuses the fitted
  encoders/scaler, returns a churn label and probability
- **SQLite storage with full CRUD** — Create, Read, Update, Delete on stored
  prediction records
- **Analytics & reporting** — churn rate, churn rate by contract type, and
  average charges for churners vs. non-churners
- **Logging** — every module logs to `logs/app.log` and the console
- **Unit tests** — cover preprocessing, model training, and database CRUD

## Technologies / Tools Used

- Python 3
- pandas, NumPy — data handling
- scikit-learn — model training and evaluation (RandomForestClassifier)
- joblib — model/encoder/scaler persistence
- SQLite (`sqlite3`, standard library) — storage layer
- Python's built-in `logging` module

## Project Structure

```
churn-prediction-system/
├── README.md
├── statement.md
├── requirements.txt
├── src/
│   ├── config.py          # central configuration / paths / constants
│   ├── logger.py          # shared logger setup
│   ├── data_loader.py      # Module 1a: data ingestion + validation
│   ├── preprocessing.py    # Module 1b: cleaning, encoding, scaling, split
│   ├── model.py             # Module 2a: model training/evaluation
│   ├── predictor.py         # Module 2b: prediction pipeline
│   ├── database.py          # Module 3a: SQLite CRUD storage layer
│   ├── report.py            # Module 3b: analytics/reporting
│   └── app.py                # CLI entry point tying everything together
├── data/                    # sample_customers.csv is generated here
├── models/                  # trained model + encoders + scaler (generated)
├── logs/                    # app.log (generated)
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_database.py
└── diagrams/                # design diagrams referenced in the report
```

## Steps to Install & Run

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd churn-prediction-system
   ```

2. **Install dependencies** (Python 3.9+ recommended)
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python -m src.app
   ```

4. **Use the menu:**
   - Option `1` trains the model (auto-generates a sample dataset on first run)
   - Option `2` predicts churn for a new customer you enter interactively
   - Option `3` shows the analytics report over everything predicted so far
   - Options `4`–`6` let you view, update, and delete stored prediction records

## Instructions for Testing

Each test file can be run directly (no test framework required):

```bash
python tests/test_preprocessing.py
python tests/test_model.py
python tests/test_database.py
```

Each script prints `All ... tests passed.` on success and raises an
`AssertionError` with a clear message if a check fails.

## Non-Functional Requirements Addressed

| Requirement      | How it's addressed |
|-------------------|--------------------------------------------------------|
| Performance       | Random Forest with bounded depth (`max_depth=8`) for fast train/predict; SQLite for lightweight local storage |
| Reliability       | Input validation (`DataValidationError`, `PredictionError`) prevents silent bad predictions |
| Usability         | Simple numbered CLI menu; clear labelled output |
| Maintainability   | One responsibility per module; all constants centralised in `config.py` |
| Scalability       | Stateless prediction function; SQLite can be swapped for PostgreSQL/MySQL by changing only `database.py` |
| Error handling    | Every module wraps risky operations in try/except with specific custom exceptions |
| Logging/Monitoring| Centralised logger writes to both console and `logs/app.log` |
| Security          | Parameterised SQL queries throughout `database.py` prevent SQL injection |

## Future Enhancements

- Replace the CLI with a web dashboard (Flask/Streamlit)
- Support CSV batch upload for bulk predictions
- Add model versioning and A/B comparison between algorithms
- Add authentication for multi-user deployments
