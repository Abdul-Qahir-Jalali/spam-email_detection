# 📧 End-to-End MLOps Spam Detection System

![CI Status](https://github.com/Abdul-Qahir-Jalali/spam-email-detection/actions/workflows/ci.yaml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)

A production-grade **Machine Learning Operations (MLOps)** pipeline that detects spam emails. 

Unlike standard ML projects, this system focuses on **lifecycle automation**. It features a "Self-Healing" CI/CD pipeline that automatically detects data updates, retrains the model, verifies performance, and packages the application without manual intervention.

---

## 🏗️ Architecture & Tech Stack

This project implements a robust MLOps workflow:

* **Language:** Python 3.9
* **Model:** Naive Bayes (Scikit-Learn)
* **API Serving:** FastAPI
* **Containerization:** Docker
* **Data Versioning:** DVC (Data Version Control)
* **Experiment Tracking:** MLflow
* **Automation (CI/CD):** GitHub Actions
* **Testing:** Pytest

### The Pipeline Flow
`Raw Data` ➔ `Preprocessing` ➔ `Training (MLflow)` ➔ `Evaluation` ➔ `Testing` ➔ `Docker Build`

---

## 🚀 Key Features

* **🔄 Automated Retraining:** Pushing new data to GitHub triggers the full training pipeline.
* **🛡️ Data Drift Handling:** The system can "heal" itself. If the model fails on new spam trends (e.g., Crypto scams), adding data automatically produces a fixed model.
* **📦 Containerized:** Runs identically on any machine using Docker.
* **📊 Experiment Tracking:** Logs Accuracy, Precision, and model parameters using MLflow.
* **⚡ High-Performance API:** Real-time inference using FastAPI.

---

## 📂 Project Structure

```bash
spam-email-detection/
├── .github/workflows/    # CI/CD Pipeline (GitHub Actions)
├── data/                 # Raw and Processed Data (Managed by DVC)
├── models/               # Trained Models (.pkl)
├── src/                  # Source Code
│   ├── preprocess.py     # Data cleaning & splitting
│   └── train.py          # Training script with MLflow logging
├── tests/                # Automated Tests
│   └── test_app.py       # API & Model integration tests
├── Dockerfile            # Blueprint for the container
├── requirements.txt      # Python dependencies
└── app.py                # FastAPI Application
```

## 🛠️ Installation & Usage

### Option 1: Run with Docker (Recommended)
You don't need to install Python or libraries. Just need Docker.

Build the Image:

```bash
docker build -t spam-detector:latest .
```

Run the Container:

```bash
docker run -p 8000:8000 spam-detector:latest
```

Test the API: Open your browser to: http://127.0.0.1:8000/docs

### Option 2: Run Locally
Clone the repository:

```bash
git clone https://github.com/Abdul-Qahir-Jalali/spam-email-detection.git
cd spam-email-detection
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

Run the Training Pipeline:

```bash
python src/preprocess.py
python src/train.py
```

Start the Server:

```bash
uvicorn app:app --reload
```

## 🧪 The "Self-Healing" Scenario (Demo)
This project was stress-tested against Data Drift.

The Attack: The initial model was trained on old data. When tested with a modern Crypto Scam message ("Invest in Bitcoin today..."), the model failed and classified it as Ham (Safe).

The Fix: Instead of changing the code, I acted as a Data Engineer:

1. Added the specific Bitcoin spam example to `data/spam.csv`.
2. Pushed the change to GitHub.

The Automation:

1. GitHub Actions detected the data change.
2. The robot automatically woke up, retrained the model, and ran a specific verification test.

Result: The new model correctly identified the Crypto scam as Spam.

See the Verification Log in `src/train.py`:

```plaintext
>>> STEP 3: VERIFYING THE FIX (The Bitcoin Test)
📝 Test Phrase: 'Invest in Bitcoin today for huge returns. Crypto is the future.'
🤖 Prediction:  [SPAM]
✅ RESULT: FIX VERIFIED! The model now catches crypto scams.
```

## 📊 API Reference
Endpoint: `POST /predict`

Request Body:

```json
{
  "text": "Congratulations! You have won a free lottery ticket. Click here."
}
```

Response:

```json
{
  "text": "Congratulations! You have won a free lottery ticket. Click here.",
  "prediction": "Spam",
  "probability": 0.98
}
```

## 📈 Experiment Tracking (MLflow)
To view training history and metrics:

Run the command:

```bash
mlflow ui
```

Open http://127.0.0.1:5000 (or the port specified in your terminal).

## 👤 Author
Abdul Qahir Jalali
