from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 1. Start the App
app = FastAPI()

# 2. Load the Brain (The Model)
# We load it right away so it is ready for requests
print("Loading model...")
model = joblib.load('models/model.pkl')

# 3. Define the Rules (The Input Schema)
# This tells FastAPI: "I only accept data that looks like this."
class EmailRequest(BaseModel):
    text: str

# 4. Create the "Reception Desk" (The Endpoint)
@app.post("/predict")
def predict_spam(email: EmailRequest):
    # The 'email' variable holds the data the user sent
    
    # Ask the robot
    prediction_number = model.predict([email.text])[0]
    
    # Translate number to words
    label = "Spam" if prediction_number == 1 else "Ham"
    
    return {
        "text": email.text, 
        "prediction": label
    }