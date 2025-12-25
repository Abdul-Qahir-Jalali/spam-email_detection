import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
import joblib
import mlflow

# 1. Start MLflow Run
mlflow.set_experiment("Spam_Detector")

with mlflow.start_run():
    # 2. Load Data
    print("Loading data...")
    df = pd.read_csv('data/processed/train.csv')
    
    # We actually need to split again to get a "Validation" set for scoring
    # (In real life we use the test.csv, but for simplicity we split here)
    X = df['text'].fillna('')
    y = df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # 3. Create Pipeline
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # 4. Train
    print("Training model...")
    model.fit(X_train, y_train)

    # 5. Evaluate
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")

    # 6. Log Metrics to MLflow (The Magic Part)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    
    # Log parameters (What algorithm did we use?)
    mlflow.log_param("algorithm", "NaiveBayes")

    # 7. Save Model
    joblib.dump(model, 'models/model.pkl')
    # Optional: Log the model file itself to MLflow
    mlflow.log_artifact('models/model.pkl')

    print("Model trained and metrics logged to MLflow!")