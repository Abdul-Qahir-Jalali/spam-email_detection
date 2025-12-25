import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json

# 1. Load the Model and the Exam Data
print("Loading model and test data...")
model = joblib.load('models/model.pkl')
test_df = pd.read_csv('data/processed/test.csv')

# Handle missing values
test_df['text'] = test_df['text'].fillna('')

X_test = test_df['text']
y_test = test_df['label']

# 2. Make Predictions (Take the Exam)
print("Making predictions...")
predictions = model.predict(X_test)

# 3. Grade the Exam
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

# 4. Show the Report Card
print("--------------------------------")
print(f"Accuracy:  {accuracy:.2f}")  # e.g., 0.98 means 98%
print(f"Precision: {precision:.2f}") # e.g., 1.00 means 100% correct when detecting spam
print(f"Recall:    {recall:.2f}")    # e.g., 0.90 means it caught 90% of the spam
print("--------------------------------")

# 5. Save the Grades (for DVC to track later)
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to metrics.json")