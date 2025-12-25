import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. Settings
# The raw data doesn't have headers, so we define them manually
columns = ['label', 'text']
# The file is actually separated by "tabs", not commas
input_file = 'data/spam.csv' 
output_folder = 'data/processed'

# 2. Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# 3. Load the data
print("Loading data...")
# sep='\t' tells pandas to look for tabs, header=None says "there are no titles"
df = pd.read_csv(input_file, sep='\t', names=columns, header=None)

print(f"Total emails found: {len(df)}")

# 4. Simple Preprocessing (Convert "ham"/"spam" to numbers)
# Machine learning likes numbers, not words.
# spam = 1, ham (safe) = 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# 5. Split the data
# 80% for Training, 20% for Testing
print("Splitting data...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 6. Save the new files
train_df.to_csv(f'{output_folder}/train.csv', index=False)
test_df.to_csv(f'{output_folder}/test.csv', index=False)

print("Done! Files saved to data/processed/")