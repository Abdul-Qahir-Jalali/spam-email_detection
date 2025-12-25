import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# 1. Load the "Study Materials"
print("Loading training data...")
train_df = pd.read_csv('data/processed/train.csv')

# Handle missing values (just in case)
train_df['text'] = train_df['text'].fillna('')

# 2. Separate the Questions (X) from Answers (y)
X_train = train_df['text']
y_train = train_df['label']

# 3. Create the "Robot Pipeline"
# Step A: CountVectorizer turns words into numbers
# Step B: MultinomialNB is the brain that learns
print("Training the model...")
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# 4. Train the robot
model.fit(X_train, y_train)

# 5. Save the brain to a file
print("Saving the model...")
joblib.dump(model, 'models/model.pkl')

print("Done! Model saved to models/model.pkl")