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

print("\n" + "="*50)
print(">>> STEP 1: CHECKING DATA FOR NEW ISSUES")
print("="*50)

# Load Data
df = pd.read_csv('data/processed/train.csv')
print(f"Total Training Examples: {len(df)}")

# Verify if our "Bitcoin" fix is actually in the data
bitcoin_check = df[df['text'].str.contains("Bitcoin", case=False)]
if not bitcoin_check.empty:
    print("✅ SUCCESS: Found 'Bitcoin' samples in the training data!")
    print(f"   Count: {len(bitcoin_check)}")
else:
    print("❌ WARNING: 'Bitcoin' data NOT found. Model might fail.")


# 2. Split Data
X = df['text'].fillna('')
y = df['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    print("\n" + "="*50)
    print(">>> STEP 2: TRAINING NEW MODEL")
    print("="*50)
    
    # Create Pipeline
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)

    print(f"📊 New Model Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)

    # Save Model
    joblib.dump(model, 'models/model.pkl')

    # ---------------------------------------------------------
    # STEP 3: THE FINAL PROOF (Testing the Fix)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(">>> STEP 3: VERIFYING THE FIX (The Bitcoin Test)")
    print("="*50)
    
    test_phrase = "Invest in Bitcoin today for huge returns. Crypto is the future."
    pred_code = model.predict([test_phrase])[0]
    pred_label = "Spam" if pred_code == 1 else "Ham"
    
    print(f"📝 Test Phrase: '{test_phrase}'")
    print(f"🤖 Prediction:  [{pred_label.upper()}]")
    
    if pred_label == "Spam":
        print("\n✅ RESULT: FIX VERIFIED! The model now catches crypto scams.")
    else:
        print("\n❌ RESULT: FAILED. The model still thinks this is safe.")
    
    print("="*50 + "\n")